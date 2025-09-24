#!/bin/bash

# Simple combination approach using existing agents only
set -e
set -x

# --- Configuration ---
LLM_NAME="gpt-4o-mini"
DOMAIN="humaneval"

# Simple combination: Use existing agents with more Programming_Expert agents
# 2 Programming_Expert agents (for multiple code generation attempts)
# 1 Test_Analyst (for test analysis)
# 1 Bug_Fixer (for bug fixing)
# 1 Algorithm_Designer (for algorithm design)
AGENT_NAMES="Programming_Expert Programming_Expert Test_Analyst Bug_Fixer Algorithm_Designer"
AGENT_NUMS="2 1 1 1" # 2 Programming_Experts, 1 of each other agent
BATCH_SIZE=2
DATASET_JSON="datasets/humaneval/humaneval-py.jsonl"
OUTPUT_DIR="trained_models"

# Ensure the output directory exists
mkdir -p $OUTPUT_DIR

BASE_CMD="python -m experiments.run_humaneval --llm_name $LLM_NAME --domain $DOMAIN --agent_names $AGENT_NAMES --agent_nums $AGENT_NUMS --dataset_json $DATASET_JSON"

# --- GTD Mode with Simple Combination Approach ---
# This mode uses a Graph Topology Diffusion model to generate agent communication graphs.
# It consists of three phases with existing agents.

# == Phase 1: Generate initial dataset for GTD models ==
echo "--- Running GTD Phase 1: Dataset Generation with Simple Combination Approach ---"
echo "Agents: 2x Programming_Expert + Test_Analyst + Bug_Fixer + Algorithm_Designer"
$BASE_CMD \
     --mode GTD \
     --gtd-generate-data \
     --gtd-datagen-limit 10 \
     --gtd-dataset-path "$OUTPUT_DIR/gtd_humaneval_simple_combination_dataset.jsonl"

# == Phase 2: Train Proxy and Diffusion models ==
echo "--- Running GTD Phase 2: Model Training with Simple Combination Approach ---"
$BASE_CMD \
    --mode GTD \
    --gtd-train-models \
    --gtd-epochs 10 \
    --gtd-dataset-path "$OUTPUT_DIR/gtd_humaneval_simple_combination_dataset.jsonl" \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_humaneval_simple_combination.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_humaneval_simple_combination.pth"

# == Phase 3: Run inference with a pre-trained GTD Framework ==
echo "--- Running GTD Phase 3: Inference with Simple Combination Approach ---"
$BASE_CMD \
    --mode GTD \
    --batch_size $BATCH_SIZE \
    --gtd-proxy-model-path "$OUTPUT_DIR/proxy_model_humaneval_simple_combination.pth" \
    --gtd-diffusion-model-path "$OUTPUT_DIR/diffusion_model_humaneval_simple_combination.pth"

echo "--- Simple Combination Agent Pipeline Complete (All 3 Phases) ---"
