#!/usr/bin/env python3
"""
Unified CoT Generation + Hidden State Extraction Pipeline

Faithfully follows the two-pass approach from the reasoning-trajectory repo:
  - Pass 1: fast generation via model.generate()
  - Pass 2: single forward pass over the full sequence with output_hidden_states=True

For each "Step k:" marker and "####" answer marker in the generated sequence,
extracts the hidden state at (pos - 1) across all requested layers — the
representation of the token immediately preceding that anchor, i.e. the
state that predicted it.

Handles:
  - ReliableMath (unsol/sol parquet) — with solvability tags
  - GSM8K / AIME — standard QA format
  - Thinking models (<think>...</think>) — Step anchors classified as "thinking",
    #### anchor as "response"
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import random
import re

# Block all HuggingFace network calls — safe on air-gapped HPC nodes.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# -------------------------
# Arguments
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-pass CoT generation + hidden state extraction"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=["reliablemath_unsol", "reliablemath_sol", "gsm8k", "aime"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="Layer indices to extract (0 = embedding, 1..N = transformer layers)")
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--thinking_model", action="store_true",
                        help="Model wraps reasoning in <think>...</think>")
    parser.add_argument("--force_think_prefix", action="store_true",
                        help="Append '<think>\\n' to prompt for thinking models")

    parser.add_argument("--save_float16", action="store_true",
                        help="Save hidden states as float16 (halves disk usage)")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Checkpoint frequency (examples)")

    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--num_examples", type=int, default=None)

    # TTS (True Thinking Score)
    parser.add_argument("--compute_tts", action="store_true",
                        help="Compute TTS scores per step (requires 4 extra forward passes per step)")
    parser.add_argument("--tts_threshold_low", type=float, default=0.0,
                        help="Steps with TTS <= this are classified as decorative (default: 0.0)")
    parser.add_argument("--tts_threshold_high", type=float, default=0.9,
                        help="Steps with TTS > this are classified as true thinking (default: 0.9)")
    parser.add_argument("--tts_early_exit_suffix", type=str, default="\n####",
                        help="Suffix appended to the prefix to trigger early-exit answer generation "
                             "(default '\\n####' for GSM8K; use '\\n The final answer is \\\\boxed{' for MATH)")
    parser.add_argument("--tts_max_new_tokens", type=int, default=20,
                        help="Max tokens to generate during TTS early-exit (default: 20)")
    parser.add_argument("--tts_random_seed", type=int, default=42,
                        help="Random seed for number replacement in TTS perturbation (default: 42)")

    return parser.parse_args()


# -------------------------
# Utilities
# -------------------------

def get_torch_dtype(dtype_str: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]


def find_subseq(seq: List[int], subseq: List[int], start: int = 0) -> int:
    """Return first index in seq[start:] where subseq starts, or -1."""
    n, m = len(seq), len(subseq)
    if m == 0:
        return -1
    for i in range(start, n - m + 1):
        if seq[i : i + m] == subseq:
            return i
    return -1


def find_all_subseq(seq: List[int], subseq: List[int], stop_before: int = -1) -> List[int]:
    """Return all start positions of subseq in seq, optionally stopping before stop_before."""
    positions, start = [], 0
    end = stop_before if stop_before >= 0 else len(seq)
    while start < end:
        pos = find_subseq(seq, subseq, start)
        if pos == -1 or pos >= end:
            break
        positions.append(pos)
        start = pos + 1
    return positions


def build_cot_prompt(question: str) -> str:
    """Exact 'cot' prompt template from the reasoning-trajectory repo."""
    return (
        'You are a helpful assistant that solves problems step by step with each step signified by "Step [step_number]: ".\n'
        "Always provide your final answer after #### at the end.\n"
        "\n"
        f"Question: {question}\n"
        "\n"
        'Please solve this step by step, putting each step after "Step [step_number]: " and always provide your final answer after ####.\n'
        "\n"
        "Solution:\n"
        "\n"
    )


# -------------------------
# TTS (True Thinking Score)
# -------------------------

def randomly_replace_numbers(text: str, seed: Optional[int] = None) -> str:
    """Replace every number in text with a random different number.

    Matches the perturbation used in the identify_true_decorative_thinking repo
    (replace_all_numbers=True, mode='random_add_small').  For each matched
    number a small random offset in [-20, -1] ∪ [1, 20] is added so the
    replacement is always different from the original.
    """
    rng = random.Random(seed)
    number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?")

    def _replace(m: re.Match) -> str:
        original = m.group(0)
        try:
            val = float(original)
            offset = rng.choice([-1, 1]) * rng.randint(1, 20)
            new_val = val + offset
            return str(int(new_val)) if "." not in original else f"{new_val:.2f}"
        except ValueError:
            return original

    return number_pattern.sub(_replace, text)


def _tts_compare_answers(extracted: str, ground_truth: str) -> bool:
    """Lightweight answer comparison for TTS (mirrors utils_tts.compare_answers)."""
    extracted = str(extracted).split("$")[0].split("\n")[0]
    for ch in [",", ":", "%"]:
        extracted = extracted.replace(ch, "")
    extracted = extracted.strip()
    ground_truth = str(ground_truth).strip()
    if re.sub(r"\s+", "", extracted) == re.sub(r"\s+", "", ground_truth):
        return True
    try:
        if abs(float(extracted) - float(ground_truth)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    try:
        if "/" in extracted:
            n, d = extracted.split("/")
            if abs(float(n) / float(d) - float(ground_truth)) < 1e-6:
                return True
    except (ValueError, ZeroDivisionError, TypeError):
        pass
    return False


@torch.no_grad()
def _early_exit_prob(
    model,
    tokenizer,
    prefix_text: str,
    ground_truth: str,
    early_exit_suffix: str,
    max_new_tokens: int,
) -> Tuple[str, float, float]:
    """Run early-exit generation from a prefix and measure answer probabilities.

    Mirrors generate_and_analyze_checkpoint from inference_checkpoint_analysis.py.

    The model is given `prefix_text + early_exit_suffix` and asked to generate
    up to `max_new_tokens` tokens greedily.  Two probabilities are tracked:

    - extracted_prob: product of the probabilities of the tokens the model
      actually generated (only used when the model generates the correct answer)
    - ans_prob_multi: product of the probabilities of the correct-answer token(s)
      measured at each generation step while the answer is still being "expected"

    The effective probability returned is:
      - extracted_prob  if the model generated the correct answer
      - ans_prob_multi  otherwise

    This matches the logic in perturb_ee_score.
    """
    full_prompt = prefix_text + early_exit_suffix
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=4096)
    ids = inputs["input_ids"].to(model.device)
    mask = inputs["attention_mask"].to(model.device)

    ans_token_ids: List[int] = tokenizer.encode(str(ground_truth), add_special_tokens=False)
    ans_idx = 0

    generated_tokens: List[str] = []
    generated_probs: List[float] = []
    ans_token_probs: List[float] = []

    for _ in range(max_new_tokens):
        out = model(ids, attention_mask=mask)
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)

        # Track probability of the next correct-answer token
        if ans_idx < len(ans_token_ids):
            ans_token_probs.append(probs[ans_token_ids[ans_idx]].item())
            ans_idx += 1

        next_id = torch.argmax(probs).item()
        next_text = tokenizer.decode([next_id])
        next_prob = probs[next_id].item()

        # Stop at closing brace (MATH format) or newline
        if "}" in next_text or "\n" in next_text:
            break

        generated_tokens.append(next_text)
        generated_probs.append(next_prob)

        ids = torch.cat([ids, torch.tensor([[next_id]], device=model.device)], dim=1)
        mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=model.device)], dim=1)

    extracted_answer = "".join(generated_tokens).strip()

    extracted_prob = 1.0
    for p in generated_probs:
        extracted_prob *= p

    ans_prob_multi = 1.0
    for p in ans_token_probs:
        ans_prob_multi *= p

    # Use extracted_prob if the model got it right, else ans_prob_multi
    if _tts_compare_answers(extracted_answer, ground_truth):
        effective_prob = extracted_prob
    else:
        effective_prob = ans_prob_multi

    return extracted_answer, extracted_prob, ans_prob_multi


def compute_tts_scores(
    model,
    tokenizer,
    prompt: str,
    anchor_steps: List[Dict[str, Any]],
    generated_ids: List[int],
    ground_truth: str,
    early_exit_suffix: str,
    max_new_tokens: int = 20,
    a: float = 0.5,
    b: float = 0.5,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Compute True Thinking Score (TTS) for each step anchor.

    For each step i the method builds four prefix variants by independently
    perturbing (replacing all numbers) the context before step i and step i itself:

        no_perturb  : prompt + context(1..i)   + step(i)
        perturb_s   : prompt + context(1..i)   + perturb(step(i))
        perturb_c   : prompt + perturb(ctx)    + step(i)
        perturb_sc  : prompt + perturb(ctx)    + perturb(step(i))

    Each prefix is extended with `early_exit_suffix` and fed to the model for
    short greedy generation.  The effective probability of the correct answer is
    measured for each variant (matching perturb_ee_score in utils_tts.py).

        ATE(1) = |p(no_perturb) - p(perturb_s)|   # effect of perturbing step
        ATE(0) = |p(perturb_sc) - p(perturb_c)|   # same, but over a perturbed context
        TTS    = a * ATE(1) + b * ATE(0)           # default: a=b=0.5

    Returns a list of dicts (one per step anchor), aligned with the "step"-type
    entries in `anchor_steps`.
    """
    tts_results: List[Dict[str, Any]] = []

    for anchor_idx, anchor in enumerate(anchor_steps):
        if anchor["anchor_type"] != "step":
            continue

        rel_pos = anchor["rel_pos"]
        # End of this step's span = start of the next anchor (or end of sequence)
        next_rel = (
            anchor_steps[anchor_idx + 1]["rel_pos"]
            if anchor_idx + 1 < len(anchor_steps)
            else len(generated_ids)
        )

        context_ids = generated_ids[:rel_pos]
        step_ids = generated_ids[rel_pos:next_rel]

        context_text = tokenizer.decode(context_ids, skip_special_tokens=True)
        step_text = tokenizer.decode(step_ids, skip_special_tokens=True)

        context_perturbed = randomly_replace_numbers(context_text, seed=random_seed)
        step_perturbed = randomly_replace_numbers(step_text, seed=random_seed)

        variants = {
            "no_perturb": prompt + context_text + step_text,
            "perturb_s":  prompt + context_text + step_perturbed,
            "perturb_c":  prompt + context_perturbed + step_text,
            "perturb_sc": prompt + context_perturbed + step_perturbed,
        }

        probs: Dict[str, float] = {}
        for key, prefix in variants.items():
            extracted, ext_prob, ans_multi = _early_exit_prob(
                model, tokenizer, prefix, ground_truth, early_exit_suffix, max_new_tokens
            )
            # Use extracted_prob if model got it right, else ans_prob_multi
            # (mirrors perturb_ee_score in utils_tts.py)
            probs[key] = ext_prob if _tts_compare_answers(extracted, ground_truth) else ans_multi

        ate1 = abs(probs["no_perturb"] - probs["perturb_s"])
        ate0 = abs(probs["perturb_sc"] - probs["perturb_c"])
        tts = a * ate1 + b * ate0

        tts_results.append({
            "anchor_idx": anchor_idx,
            "rel_pos": rel_pos,
            "tts": float(tts),
            "ate1": float(ate1),
            "ate0": float(ate0),
            "p_no_perturb": float(probs["no_perturb"]),
            "p_perturb_s": float(probs["perturb_s"]),
            "p_perturb_c": float(probs["perturb_c"]),
            "p_perturb_sc": float(probs["perturb_sc"]),
        })

    return tts_results


# -------------------------
# Two-pass generation + hidden state extraction
# -------------------------

@torch.no_grad()
def generate_twopass_with_hidden_states(
    model,
    tokenizer,
    prompt_text: str,
    layers: List[int],
    step_token_seq: List[int],
    hash_token_seq: List[int],
    close_think_seq: List[int],
    max_input_length: int,
    max_new_tokens: int,
    temperature: float = 0.0,
    do_sample: bool = False,
    top_p: float = 1.0,
    thinking_model: bool = False,
) -> Tuple[str, List[int], List[Dict[str, Any]], Dict[int, Dict[str, List[torch.Tensor]]]]:
    """
    Two-pass generation with two hidden-state extraction approaches.

    Pass 1 — fast generation:
        model.generate() produces the full token sequence without capturing activations.

    Pass 2 — single forward pass:
        One forward pass over the complete (prompt + generated) sequence with
        output_hidden_states=True, from which both approaches are computed:

        "marker" — h^(l)_{t(Step k) - 1}:
            The hidden state at (full_pos - 1), i.e. the token immediately preceding
            the anchor.  This is the representation that predicted the anchor token,
            exactly as in the reasoning-trajectory repo.

        "mean" — mean over the step span:
            The mean of hidden states across all token positions in the step span
            [rel_pos, next_rel_pos), i.e. all tokens belonging to this step.

        Both are extracted from the same forward pass at no extra cost.

    Anchors detected (in the generated portion only):
        - "Step" token sequence → type "step", section "thinking" or "response"
        - "####" token sequence → type "answer", section "response"

    Returns:
        generated_text        : decoded generated text
        generated_ids         : list[int] of generated token IDs
        anchor_steps          : list of dicts, one per anchor, in position order
        per_layer_step_hiddens: {layer: {"marker": [...], "mean": [...]}} aligned with anchor_steps
    """
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    prompt_length = input_ids.shape[1]

    # --- Pass 1: fast generation (no hidden state capture) ---
    gen_kwargs: Dict[str, Any] = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs["do_sample"] = False

    full_ids = model.generate(**gen_kwargs)          # [1, prompt_len + gen_len]
    generated_ids: List[int] = full_ids[0, prompt_length:].tolist()

    if not generated_ids:
        return "", [], [], {layer: {"marker": [], "mean": []} for layer in layers}

    # --- Pass 2: single forward pass over the full sequence ---
    full_attn = torch.ones(1, full_ids.shape[1], dtype=torch.long, device=model.device)
    fwd = model(
        input_ids=full_ids,
        attention_mask=full_attn,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    # fwd.hidden_states: tuple of (num_layers + 1) tensors, each [1, seq_len, hidden_dim]
    # Index 0 = embedding output; indices 1..N = transformer layer outputs.

    # --- Locate </think> boundary (relative to generated sequence) ---
    think_close_rel = -1
    if thinking_model and close_think_seq:
        think_close_rel = find_subseq(generated_ids, close_think_seq)

    # --- Locate all "Step" anchors ---
    # For thinking models: only before </think>.
    # For standard models: across the entire generated sequence.
    step_search_end = think_close_rel if (thinking_model and think_close_rel >= 0) else len(generated_ids)
    step_positions = find_all_subseq(generated_ids, step_token_seq, stop_before=step_search_end)

    # --- Locate "####" answer anchor ---
    hash_search_start = 0
    if thinking_model and think_close_rel >= 0:
        hash_search_start = think_close_rel + len(close_think_seq)
    hash_pos = find_subseq(generated_ids, hash_token_seq, hash_search_start)

    # --- Build ordered anchor list: (rel_pos, anchor_type, section) ---
    anchor_positions: List[Tuple[int, str, str]] = []
    for sp in step_positions:
        section = "thinking" if (thinking_model and think_close_rel >= 0) else "response"
        anchor_positions.append((sp, "step", section))
    if hash_pos >= 0:
        anchor_positions.append((hash_pos, "answer", "response"))
    anchor_positions.sort(key=lambda x: x[0])

    if not anchor_positions:
        logger.warning("No Step markers or #### found in generated sequence.")
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text, generated_ids, [], {layer: {"marker": [], "mean": []} for layer in layers}

    # --- Extract both hidden-state approaches for each anchor ---
    anchor_steps: List[Dict[str, Any]] = []
    per_layer_step_hiddens: Dict[int, Dict[str, List[torch.Tensor]]] = {
        layer: {"marker": [], "mean": []} for layer in layers
    }

    for i, (rel_pos, anchor_type, section) in enumerate(anchor_positions):
        full_pos = prompt_length + rel_pos
        hidden_pos = full_pos - 1          # token immediately preceding the anchor
        if hidden_pos < 0:
            continue

        # Step span: from this anchor up to (not including) the next anchor
        next_rel = anchor_positions[i + 1][0] if i + 1 < len(anchor_positions) else len(generated_ids)
        step_text = tokenizer.decode(
            generated_ids[rel_pos:next_rel], skip_special_tokens=False
        ).strip()

        anchor_steps.append({
            "rel_pos": rel_pos,
            "hidden_pos": hidden_pos,
            "anchor_type": anchor_type,
            "section": section,
            "step_text": step_text,
            "token_count": next_rel - rel_pos,
        })

        span_start = prompt_length + rel_pos      # inclusive, in full-sequence coords
        span_end   = prompt_length + next_rel     # exclusive

        for layer in layers:
            if layer >= len(fwd.hidden_states):
                raise ValueError(
                    f"Requested layer {layer} but model only has {len(fwd.hidden_states)} hidden states."
                )
            hs = fwd.hidden_states[layer]         # [1, seq_len, hidden_dim]

            # Approach 1 — marker: hidden state at (full_pos - 1)
            h_marker = hs[0, hidden_pos, :].detach().cpu()
            per_layer_step_hiddens[layer]["marker"].append(h_marker)

            # Approach 2 — mean: average over all token positions in the step span
            h_mean = hs[0, span_start:span_end, :].mean(dim=0).detach().cpu()
            per_layer_step_hiddens[layer]["mean"].append(h_mean)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, generated_ids, anchor_steps, per_layer_step_hiddens


# -------------------------
# Dataset loading
# -------------------------

def load_dataset_by_type(dataset_type: str, dataset_path: str, split: str = "test"):
    """Load dataset entirely from local files — no network access."""
    logger.info(f"Loading {dataset_type} from {dataset_path}")
    p = Path(dataset_path)

    if p.is_dir():
        # Directory: glob for parquet or json files, sorted for determinism.
        parquet_files = sorted(p.glob("*.parquet"))
        json_files = sorted(p.glob("*.jsonl")) or sorted(p.glob("*.json"))
        if parquet_files:
            ds = load_dataset("parquet", data_files=[str(f) for f in parquet_files])["train"]
        elif json_files:
            ds = load_dataset("json", data_files=[str(f) for f in json_files])["train"]
        else:
            raise FileNotFoundError(f"No .parquet or .jsonl files found in {dataset_path}")
    elif p.suffix in {".parquet"}:
        ds = load_dataset("parquet", data_files=str(p))["train"]
    elif p.suffix in {".jsonl", ".json"}:
        ds = load_dataset("json", data_files=str(p))["train"]
    else:
        raise ValueError(
            f"Unsupported dataset_path format: {dataset_path}. "
            "Provide a .parquet file, a .jsonl/.json file, or a directory containing them."
        )

    logger.info(f"Loaded {len(ds)} examples from {dataset_path}")
    return ds


# -------------------------
# Saving
# -------------------------

def save_metadata(metadata: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(metadata)} metadata records to {path}")


def save_hidden_states(
    steps_data: Dict[int, np.ndarray],
    output_dir: Path,
    dataset_tag: str,
    section: str,
    approach: str,
    layer: int,
    start_idx: int,
    end_idx: Optional[int],
    save_float16: bool,
) -> None:
    """Save one hidden-state vector per anchor step to an NPZ file.

    approach is "marker" (pos-1) or "mean" (span average).
    Array shape: [num_steps, hidden_dim].
    """
    out_dtype = np.float16 if save_float16 else np.float32
    chunk_end = end_idx if end_idx is not None else "end"

    sorted_keys = sorted(steps_data.keys())
    vecs = [steps_data[k] for k in sorted_keys]
    arr = np.array(vecs, dtype=out_dtype) if vecs else np.zeros((0, 0), dtype=out_dtype)

    fname = output_dir / f"cot_{dataset_tag}_{start_idx}_{chunk_end}_{section}_{approach}_layer{layer}.npz"
    np.savez(str(fname),
             hidden_states=arr,
             step_indices=np.array(sorted_keys, dtype=np.int32))
    logger.info(f"Saved {len(vecs)} steps | section={section} approach={approach} layer={layer} → {fname}")


# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset bookkeeping
    if "reliablemath" in args.dataset_type:
        dataset_tag = "reliablemath"
        hardcoded_solvability = "unsolvable" if "unsol" in args.dataset_type else "solvable"
    elif args.dataset_type == "gsm8k":
        dataset_tag = "gsm8k"
        hardcoded_solvability = None
    else:
        dataset_tag = "aime"
        hardcoded_solvability = None

    # Load model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=get_torch_dtype(args.dtype),
        device_map="auto",
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    model.eval()

    num_layers = getattr(model.config, "num_hidden_layers", None)
    logger.info(f"Model has {num_layers} transformer layers. Extracting: {args.layers}")
    if num_layers is not None:
        bad = [l for l in args.layers if l < 0 or l > num_layers]
        if bad:
            raise ValueError(f"Invalid layer indices: {bad}  (valid: 0..{num_layers})")

    # Pre-compute token sequences for anchor detection
    # "Step" — the word that starts every step marker ("Step 1:", "Step 2:", ...)
    step_token_seq: List[int] = tokenizer.encode("Step", add_special_tokens=False)
    # "####" — the GSM8K answer delimiter
    hash_token_seq: List[int] = tokenizer.encode("####", add_special_tokens=False)
    # "</think>" — section boundary for thinking models
    close_think_seq: List[int] = (
        tokenizer.encode("</think>", add_special_tokens=False) if args.thinking_model else []
    )

    logger.info(f"'Step'    → token IDs {step_token_seq}")
    logger.info(f"'####'    → token IDs {hash_token_seq}")
    if args.thinking_model:
        logger.info(f"'</think>' → token IDs {close_think_seq}")

    # Load dataset
    ds = load_dataset_by_type(args.dataset_type, args.dataset_path, args.split)
    end_idx = args.end_idx if args.end_idx is not None else len(ds)
    if args.num_examples:
        end_idx = min(end_idx, args.start_idx + args.num_examples)
    ds = ds.select(range(args.start_idx, end_idx))
    logger.info(f"Processing examples {args.start_idx}–{end_idx} ({len(ds)} total)")

    # Accumulators keyed by section ("thinking" / "response")
    metadata: Dict[str, List[Dict[str, Any]]] = {"thinking": [], "response": []}
    # hidden_states[section][approach][layer][global_step_idx] = np.ndarray
    hidden_states: Dict[str, Dict[str, Dict[int, Dict[int, np.ndarray]]]] = {
        section: {
            approach: {layer: {} for layer in args.layers}
            for approach in ("marker", "mean")
        }
        for section in ("thinking", "response")
    }
    step_counters: Dict[str, int] = {"thinking": 0, "response": 0}

    for local_idx, example in enumerate(tqdm(ds, desc="Processing")):
        example_idx = args.start_idx + local_idx

        # --- Extract fields ---
        if args.dataset_type == "gsm8k":
            question = example["question"]
            ground_truth = example.get("answer", "")
            is_solvable = None
        elif args.dataset_type == "aime":
            question = example.get("problem", example.get("question", ""))
            ground_truth = example.get("answer", "")
            is_solvable = None
        else:
            question = example.get("problem", example.get("question", ""))
            ground_truth = example.get("ground_truth", example.get("answer", ""))
            is_solvable = hardcoded_solvability

        # --- Build prompt ---
        prompt = build_cot_prompt(question)
        if args.thinking_model and args.force_think_prefix:
            prompt = prompt + "<think>\n"

        # --- Generate + extract hidden states ---
        try:
            _, generated_ids, anchor_steps, per_layer_hiddens = generate_twopass_with_hidden_states(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                layers=args.layers,
                step_token_seq=step_token_seq,
                hash_token_seq=hash_token_seq,
                close_think_seq=close_think_seq,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                top_p=args.top_p,
                thinking_model=args.thinking_model,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at example {example_idx}, skipping")
                torch.cuda.empty_cache()
                continue
            raise

        if not anchor_steps:
            logger.warning(f"Example {example_idx}: no anchors detected, skipping")
            continue

        # --- TTS computation (optional, only for GSM8K/AIME, not ReliableMath) ---
        # Build a lookup from anchor_idx → TTS dict so we can merge into metadata below.
        tts_by_anchor: Dict[int, Dict[str, float]] = {}
        should_compute_tts = (
            args.compute_tts
            and "reliablemath" not in args.dataset_type
            and ground_truth
        )
        if should_compute_tts:
            try:
                tts_list = compute_tts_scores(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    anchor_steps=anchor_steps,
                    generated_ids=generated_ids,
                    ground_truth=str(ground_truth),
                    early_exit_suffix=args.tts_early_exit_suffix,
                    max_new_tokens=args.tts_max_new_tokens,
                    a=0.5,
                    b=0.5,
                    random_seed=args.tts_random_seed,
                )
                for t in tts_list:
                    tts_by_anchor[t["anchor_idx"]] = t
            except Exception as e:
                logger.warning(f"TTS failed for example {example_idx}: {e}")

        # --- Store each anchor's hidden state and metadata ---
        for anchor_idx, anchor in enumerate(anchor_steps):
            section = anchor["section"]
            global_step_idx = step_counters[section]

            tts_entry = tts_by_anchor.get(anchor_idx, {})
            meta_record: Dict[str, Any] = {
                "step_global_idx": global_step_idx,
                "example_idx": example_idx,
                "section": section,
                "anchor_type": anchor["anchor_type"],   # "step" or "answer"
                "step_text": anchor["step_text"],
                "rel_pos": anchor["rel_pos"],
                "hidden_pos": anchor["hidden_pos"],
                "token_count": anchor["token_count"],
                "question": question,
                "ground_truth": ground_truth,
                "is_solvable": is_solvable,
            }
            if tts_entry:
                meta_record["tts"] = tts_entry["tts"]
                meta_record["ate1"] = tts_entry["ate1"]
                meta_record["ate0"] = tts_entry["ate0"]
                meta_record["p_no_perturb"] = tts_entry["p_no_perturb"]
                meta_record["p_perturb_s"] = tts_entry["p_perturb_s"]
                meta_record["p_perturb_c"] = tts_entry["p_perturb_c"]
                meta_record["p_perturb_sc"] = tts_entry["p_perturb_sc"]
                # Classification matching tts.py thresholds
                tts_val = tts_entry["tts"]
                if tts_val <= args.tts_threshold_low:
                    meta_record["tts_label"] = "decorative"
                elif tts_val > args.tts_threshold_high:
                    meta_record["tts_label"] = "true_thinking"
                else:
                    meta_record["tts_label"] = "ambiguous"
            metadata[section].append(meta_record)

            for layer in args.layers:
                for approach in ("marker", "mean"):
                    h = per_layer_hiddens[layer][approach][anchor_idx].float().numpy()
                    hidden_states[section][approach][layer][global_step_idx] = h

            step_counters[section] += 1

        # --- Periodic checkpoint ---
        if (local_idx + 1) % args.save_every == 0:
            logger.info(
                f"Checkpoint {local_idx + 1}/{len(ds)} | "
                f"thinking={step_counters['thinking']} response={step_counters['response']}"
            )
            _flush(metadata, hidden_states, output_dir, dataset_tag,
                   args.start_idx, end_idx, args.layers, args.save_float16,
                   args.tts_threshold_low, args.tts_threshold_high)

    # --- Final save ---
    logger.info("=== Final save ===")
    _flush(metadata, hidden_states, output_dir, dataset_tag,
           args.start_idx, end_idx, args.layers, args.save_float16)

    # Save run config
    cfg_path = output_dir / f"cot_{dataset_tag}_{args.start_idx}_{end_idx}_config.json"
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f"Done. thinking={step_counters['thinking']}  response={step_counters['response']}")
    logger.info(f"Output: {output_dir}")


def _flush(
    metadata: Dict[str, List[Dict[str, Any]]],
    hidden_states: Dict[str, Dict[str, Dict[int, Dict[int, np.ndarray]]]],
    output_dir: Path,
    dataset_tag: str,
    start_idx: int,
    end_idx: Optional[int],
    layers: List[int],
    save_float16: bool,
    tts_threshold_low: float = 0.0,
    tts_threshold_high: float = 0.9,
) -> None:
    """Write all in-memory metadata and hidden states to disk (idempotent)."""
    chunk_end = end_idx if end_idx is not None else "end"
    for section in ("thinking", "response"):
        if metadata[section]:
            save_metadata(
                metadata[section],
                output_dir / f"cot_{dataset_tag}_{start_idx}_{chunk_end}_{section}_steps.json",
            )
        for approach in ("marker", "mean"):
            for layer in layers:
                if hidden_states[section][approach][layer]:
                    save_hidden_states(
                        hidden_states[section][approach][layer],
                        output_dir, dataset_tag, section, approach, layer,
                        start_idx, end_idx, save_float16,
                    )

    # Save low/high TTS JSONL files (mirrors tts.py output: low_tts_steps / high_tts_steps)
    all_records = metadata["thinking"] + metadata["response"]
    tts_records = [r for r in all_records if "tts" in r]
    if tts_records:
        low_path  = output_dir / f"cot_{dataset_tag}_{start_idx}_{chunk_end}_low_tts_steps.jsonl"
        high_path = output_dir / f"cot_{dataset_tag}_{start_idx}_{chunk_end}_high_tts_steps.jsonl"
        with open(low_path, "w") as f:
            for r in tts_records:
                if r.get("tts_label") == "decorative":
                    f.write(json.dumps(r) + "\n")
        with open(high_path, "w") as f:
            for r in tts_records:
                if r.get("tts_label") == "true_thinking":
                    f.write(json.dumps(r) + "\n")
        logger.info(f"TTS JSONL saved: {low_path.name}, {high_path.name}")


if __name__ == "__main__":
    main()
