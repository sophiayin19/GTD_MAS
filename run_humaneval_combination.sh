#!/bin/bash

# Combination approach: Mix of new specialized agents and existing agents
# Includes all 3 phases: Data Generation, Model Training, and Inference
set -e
set -x

# --- Configuration ---
LLM_NAME="gpt-4o-mini"
DOMAIN="humaneval"

# Combination approach: Multiple code writers + specialized agents
# 2 CodeWriter agents (for multiple code generation attempts)
# 1 AnalyzeAgent (for analysis)
# 1 CodeReviewer (for code review)
# 1 TestGenerator (for test cases)
# 1 DebuggingAgent (for debugging)
AGENT_NAMES="CodeWriter CodeWriter AnalyzeAgent CodeReviewer TestGenerator DebuggingAgent"
AGENT_NUMS="2 1 1 1 1" # 2 CodeWriters, 1 of each specialized agent
BATCH_SIZE=2
DATASET_JSON="datasets/humaneval/humaneval-py.jsonl"
OUTPUT_DIR="trained_models"

# Ensure the output directory exists
mkdir -p $OUTPUT_DIR

BASE_CMD="python -m experiments.run_humaneval --llm_name $LLM_NAME --domain $DOMAIN --agent_names $AGENT_NAMES --agent_nums $AGENT_NUMS --dataset_json $DATASET_JSON"

# --- GTD Mode with Combination Approach ---
# This mode uses a Graph Topology Diffusion model to generate agent communication graphs.
# It consists of three phases with enhanced agents.

# == Phase 1: Generate initial dataset for GTD models ==
echo "--- Running GTD Phase 1: Dataset Generation with Combination Approach ---"
echo "Agents: 2x CodeWriter + AnalyzeAgent + CodeReviewer + TestGenerator + DebuggingAgent"
$BASE_CMD \
     --mode GTD \
     --gtd-generate-data \
     --gtd-datagen-limit 10 \
     --gtd-dataset-path "$OUTPUT_DIR/gtd_humaneval_combination_dataset.jsonl"

# == Phase 2: Train Proxy and Diffusion models ==
echo "--- Running GTD Phase 2: Model Training with Combination Approach ---"
$BASE_CMD \
    --mode GTD \
    --gtd-train-models \
    --gtd-epochs 10 \
    --gtd-dataset-path "$OUTPUT_DIR/gtd_humaneval_combination_dataset.jsonl" \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_humaneval_combination.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_humaneval_combination.pth"

# == Phase 3: Run inference with a pre-trained GTD Framework ==
echo "--- Running GTD Phase 3: Inference with Combination Approach ---"
$BASE_CMD \
    --mode GTD \
    --batch_size $BATCH_SIZE \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_humaneval_combination.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_humaneval_combination.pth"

echo "--- Combination Agent Pipeline Complete (All 3 Phases) ---"
