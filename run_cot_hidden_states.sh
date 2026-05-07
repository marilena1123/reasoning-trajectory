#!/bin/bash
#SBATCH --job-name=cot_hidden_states
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --account=EUHPC_A06_067
#SBATCH --output=logs/cot_hidden_states_%x_%j.out
#SBATCH --error=logs/cot_hidden_states_%x_%j.err

# Load modules
module load cuda/12.2
module load anaconda3/2023.09-0
source activate /leonardo_work/EUHPC_D33_215/step_saes/env/ssaes

# Environment setup
unset TRANSFORMERS_CACHE
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/leonardo_work/EUHPC_D33_215/step_saes/hf_cache
export HF_DATASETS_CACHE=/leonardo_work/EUHPC_D33_215/step_saes/hf_cache/datasets
mkdir -p /leonardo_work/EUHPC_D33_215/step_saes/hf_cache
mkdir -p /leonardo_work/EUHPC_D33_215/step_saes/hf_cache/datasets

cd /leonardo_work/EUHPC_D33_215/step_saes/cot_generation
mkdir -p logs

# =========================================================================
# Configuration
# =========================================================================

# Model selection
MODEL_PATH="/leonardo_work/EUHPC_D33_215/step_saes/model/llama_31_8b_instruct"
#MODEL_PATH="/leonardo_work/EUHPC_D33_215/step_saes/model/gemma-3-27b-it"
#MODEL_PATH="/leonardo_work/EUHPC_D33_216/mzoumpou/model/deepseek_r1_distill_llama8b"
#MODEL_PATH="/leonardo_work/EUHPC_D33_216/mzoumpou/model/deepseek_r1_distill_qwen_7b"

# Dataset selection: solve, unsol, gsm8k, aime
DATASET_TAG="${1:-unsol}"

# Layers to extract hidden states from (0=embedding, 1..N=transformer layers)
LAYERS="12 24 31"

# Generation settings
MAX_INPUT_LENGTH=2048
MAX_NEW_TOKENS=2048
DTYPE="bfloat16"
ATTN_IMPL="sdpa"
# Pass 1 generation backend.
# Use vLLM (faster) or plain HuggingFace batched generation.
USE_VLLM="--use_vllm"          # set to "" to fall back to HF batched generation
VLLM_TENSOR_PARALLEL=4         # number of GPUs for vLLM (match #SBATCH --gres=gpu:N)
VLLM_GPU_MEM_UTIL=0.85         # fraction of GPU memory vLLM may use

# HF fallback batch size (ignored when USE_VLLM is set).
GENERATION_BATCH_SIZE=4

# Checkpoint frequency (examples between saves)
SAVE_EVERY=100

# Whether to save hidden states as float16 (halves disk usage)
SAVE_FLOAT16=""          # set to "--save_float16" to enable

# Thinking model settings (uncomment for DeepSeek-R1 etc.)
THINKING_FLAGS=""
# THINKING_FLAGS="--thinking_model --force_think_prefix"

# =========================================================================
# Dataset-specific configuration
# =========================================================================

if [ "$DATASET_TAG" = "solve" ]; then
  DATASET_TYPE="reliablemath_sol"
  DATASET_PATH="/leonardo_work/EUHPC_D33_215/step_saes/SSAE/data/ReliableMath_parquet/solve.parquet"
  OUTPUT_DIR="/leonardo_work/EUHPC_D33_215/step_saes/cot_generation/hidden_states_and_text_data/solve_cot_llama31_8b"
  START_IDX=0
  END_IDX=312
  TTS_FLAGS=""                  # ReliableMath: no TTS, solvability tag is hardcoded instead

elif [ "$DATASET_TAG" = "unsol" ]; then
  DATASET_TYPE="reliablemath_unsol"
  DATASET_PATH="/leonardo_work/EUHPC_D33_215/step_saes/SSAE/data/ReliableMath_parquet/unsol.parquet"
  OUTPUT_DIR="/leonardo_work/EUHPC_D33_215/step_saes/cot_generation/hidden_states_and_text_data/unsol_cot_llama31_8b"
  START_IDX=0
  END_IDX=1102
  TTS_FLAGS=""                  # ReliableMath: no TTS

elif [ "$DATASET_TAG" = "gsm8k" ]; then
  DATASET_TYPE="gsm8k"
  DATASET_PATH="/leonardo_work/EUHPC_D33_215/step_saes/data/gsm8k"
  OUTPUT_DIR="/leonardo_work/EUHPC_D33_215/step_saes/cot_generation/hidden_states_and_text_data/gsm8k_cot_llama31_8b"
  START_IDX=0
  END_IDX=1319
  TTS_FLAGS="--compute_tts \
    --tts_early_exit_suffix $'\n####' \
    --tts_max_new_tokens 20 \
    --tts_threshold_low 0.0 \
    --tts_threshold_high 0.9 \
    --tts_random_seed 42"

elif [ "$DATASET_TAG" = "aime" ]; then
  DATASET_TYPE="aime"
  DATASET_PATH="/leonardo_work/EUHPC_D33_216/mzoumpou/datasets/aime25"
  OUTPUT_DIR="/leonardo_work/EUHPC_D33_215/step_saes/cot_generation/hidden_states_and_text_data/aime_cot_llama31_8b"
  START_IDX=0
  END_IDX=30
  TTS_FLAGS="--compute_tts \
    --tts_early_exit_suffix $'\n#### ' \
    --tts_max_new_tokens 20 \
    --tts_threshold_low 0.0 \
    --tts_threshold_high 0.9 \
    --tts_random_seed 42"

else
  echo "DATASET_TAG must be 'solve', 'unsol', 'gsm8k', or 'aime'"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# =========================================================================
# Logging
# =========================================================================

echo "========================================"
echo "CoT Hidden-State Extraction Pipeline"
echo "========================================"
echo "Dataset Tag:    $DATASET_TAG"
echo "Dataset Type:   $DATASET_TYPE"
echo "Dataset Path:   $DATASET_PATH"
echo "Output Dir:     $OUTPUT_DIR"
echo "Model:          $MODEL_PATH"
echo "Layers:         $LAYERS"
echo "Max Tokens:     $MAX_NEW_TOKENS"
echo "Start Index:    $START_IDX"
echo "End Index:      $END_IDX"
echo "vLLM:           ${USE_VLLM:-disabled (HF batched)}"
echo "Batch size:     $GENERATION_BATCH_SIZE (HF fallback)"
echo "TTS:            ${TTS_FLAGS:-disabled}"
echo "Thinking model: ${THINKING_FLAGS:-no}"
echo "========================================"
echo "Job Started: $(date)"
echo "========================================"

# =========================================================================
# Run pipeline
# =========================================================================

python cot_hidden_states.py \
  --model_path    "$MODEL_PATH" \
  --dataset_path  "$DATASET_PATH" \
  --dataset_type  "$DATASET_TYPE" \
  --output_dir    "$OUTPUT_DIR" \
  --start_idx     "$START_IDX" \
  --end_idx       "$END_IDX" \
  --layers        $LAYERS \
  --max_input_length "$MAX_INPUT_LENGTH" \
  --max_new_tokens   "$MAX_NEW_TOKENS" \
  --dtype         "$DTYPE" \
  --attn_implementation "$ATTN_IMPL" \
  --save_every    "$SAVE_EVERY" \
  --generation_batch_size "$GENERATION_BATCH_SIZE" \
  $USE_VLLM \
  ${USE_VLLM:+--vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL"} \
  ${USE_VLLM:+--vllm_gpu_memory_utilization "$VLLM_GPU_MEM_UTIL"} \
  $SAVE_FLOAT16 \
  $THINKING_FLAGS \
  $TTS_FLAGS

# =========================================================================
# Exit status
# =========================================================================

EXIT_CODE=$?

echo "========================================"
echo "Job Finished: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
