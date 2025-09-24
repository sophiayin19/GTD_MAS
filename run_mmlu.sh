#!/bin/bash

set -e
set -x

# --- Configuration ---
LLM_NAME="gpt-4o-mini"
DOMAIN="mmlu"
AGENT_NAMES="Knowlegable_Expert Mathematician Critic"
AGENT_NUMS="1 1 1"
BATCH_SIZE=2
DATASET_JSON="datasets/MMLU/mmlu.jsonl"
OUTPUT_DIR="trained_models"

mkdir -p $OUTPUT_DIR

BASE_CMD="python -m experiments.run_mmlu --llm_name $LLM_NAME --domain $DOMAIN --agent_names $AGENT_NAMES --agent_nums $AGENT_NUMS --dataset_json $DATASET_JSON"

# --- GTD Mode ---

# == Phase 1: Generate initial dataset ==
echo "--- Running GTD Phase 1: Dataset Generation for MMLU ---"
$BASE_CMD \
     --gtd-generate-data \
     --gtd-datagen-limit 10 \
     --gtd-dataset-path "$OUTPUT_DIR/gtd_mmlu_dataset.jsonl"


# == Phase 2: Train Proxy and Diffusion models ==
echo "--- Running GTD Phase 2: Model Training for MMLU ---"
$BASE_CMD \
    --gtd-train-models \
    --gtd-epochs 10 \
    --gtd-dataset-path "$OUTPUT_DIR/gtd_mmlu_dataset.jsonl" \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_mmlu.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_mmlu.pth"


# == Phase 3: Run inference ==
echo "--- Running GTD Phase 3: Inference for MMLU ---"
$BASE_CMD \
    --batch_size $BATCH_SIZE \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_mmlu.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_mmlu.pth"


echo "--- Script finished ---"
