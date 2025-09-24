#!/bin/bash

# Multi-Dataset GTD Training Script
# This script trains GTD models on multiple datasets independently

set -e
set -x

# --- Global Configuration ---
LLM_NAME="gpt-4o"
BATCH_SIZE=2
EPOCHS=10
DATAGEN_LIMIT=50

# Dataset configurations
declare -A DATASETS=(
    ["gsm8k"]="datasets/gsm8k/gsm8k.jsonl"
    ["humaneval"]="datasets/humaneval/humaneval.jsonl"  
    ["mmlu"]="datasets/MMLU/mmlu.jsonl"
)

declare -A AGENT_CONFIGS=(
    ["gsm8k"]="MathSolver:4"
    ["humaneval"]="CodeSolver:4"
    ["mmlu"]="KnowledgeSolver:4" 
)

# Function to parse agent config
parse_agent_config() {
    local config=$1
    echo ${config%%:*}  # agent name
    echo ${config##*:}  # agent number
}

# Function to run complete GTD pipeline for one dataset
run_gtd_pipeline() {
    local domain=$1
    local dataset_path=$2
    local agent_config=$3
    
    echo "========================================="
    echo "Starting GTD Pipeline for: $domain"
    echo "Dataset: $dataset_path"
    echo "Agent Config: $agent_config"
    echo "========================================="
    
    # Parse agent configuration
    local agent_name=$(echo $agent_config | cut -d: -f1)
    local agent_nums=$(echo $agent_config | cut -d: -f2)
    
    # Create domain-specific output paths
    local gtd_dataset_path="gtd_${domain}_dataset.jsonl"
    local proxy_model_path="proxy_model_${domain}.pth"
    local diffusion_model_path="diffusion_model_${domain}.pth"
    
    # Base command for this domain
    local BASE_CMD="python -m experiments.run_${domain} --llm_name $LLM_NAME --domain $domain --agent_names $agent_name --agent_nums $agent_nums --dataset_json $dataset_path"
    
    # Phase 1: Generate Dataset
    echo "--- Phase 1: Dataset Generation for $domain ---"
    $BASE_CMD \
        --mode GTD \
        --gtd-generate-data \
        --gtd-datagen-limit $DATAGEN_LIMIT \
        --gtd-dataset-path "$gtd_dataset_path"
    
    if [ $? -ne 0 ]; then
        echo "Error in Phase 1 for $domain. Skipping to next dataset."
        return 1
    fi
    
    # Phase 2: Train Models
    echo "--- Phase 2: Model Training for $domain ---"
    $BASE_CMD \
        --mode GTD \
        --gtd-train-models \
        --gtd-epochs $EPOCHS \
        --gtd-dataset-path "$gtd_dataset_path" \
        --gtd-proxy-model-path "$proxy_model_path" \
        --gtd-diffusion-model-path "$diffusion_model_path"
        
    if [ $? -ne 0 ]; then
        echo "Error in Phase 2 for $domain. Skipping to next dataset."
        return 1
    fi
    
    # Phase 3: Inference (Test)
    echo "--- Phase 3: Inference for $domain ---"
    # Check if separate test file exists
    local test_dataset_path="${dataset_path/train.jsonl/test.jsonl}"
    if [ ! -f "$test_dataset_path" ]; then
        test_dataset_path="$dataset_path"
        echo "Warning: Using training dataset for testing in $domain"
    fi
    
    $BASE_CMD \
        --dataset_json "$test_dataset_path" \
        --mode GTD \
        --batch_size $BATCH_SIZE \
        --gtd-proxy-model-path "$proxy_model_path" \
        --gtd-diffusion-model-path "$diffusion_model_path"
        
    if [ $? -ne 0 ]; then
        echo "Error in Phase 3 for $domain."
        return 1
    fi
    
    echo "--- Completed GTD Pipeline for $domain ---"
    echo ""
}

# Function to run baseline comparisons
run_baselines() {
    local domain=$1
    local dataset_path=$2
    local agent_config=$3
    
    echo "--- Running Baseline Comparisons for $domain ---"
    
    local agent_name=$(echo $agent_config | cut -d: -f1)
    local agent_nums=$(echo $agent_config | cut -d: -f2)
    local BASE_CMD="python -m experiments.run_${domain} --llm_name $LLM_NAME --domain $domain --agent_names $agent_name --agent_nums $agent_nums --dataset_json $dataset_path"
    
    # FullConnected Baseline
    echo "Testing FullConnected baseline for $domain"
    $BASE_CMD --mode FullConnected --batch_size $BATCH_SIZE
    
    # Chain Baseline  
    echo "Testing Chain baseline for $domain"
    $BASE_CMD --mode Chain --batch_size $BATCH_SIZE
    
    # Star Baseline
    echo "Testing Star baseline for $domain" 
    $BASE_CMD --mode Star --batch_size $BATCH_SIZE
}

# Main execution
main() {
    echo "Starting Multi-Dataset GTD Training"
    echo "Datasets to process: ${!DATASETS[@]}"
    
    # Create results directory
    mkdir -p results/
    
    # Process each dataset independently
    for domain in "${!DATASETS[@]}"; do
        dataset_path="${DATASETS[$domain]}"
        agent_config="${AGENT_CONFIGS[$domain]}"
        
        echo "Processing dataset: $domain"
        
        # Check if dataset exists
        if [ ! -f "$dataset_path" ]; then
            echo "Warning: Dataset $dataset_path not found. Skipping $domain."
            continue
        fi
        
        # Run GTD pipeline
        if run_gtd_pipeline "$domain" "$dataset_path" "$agent_config"; then
            echo "GTD pipeline completed successfully for $domain"
            
            # Run baseline comparisons
            run_baselines "$domain" "$dataset_path" "$agent_config"
        else
            echo "GTD pipeline failed for $domain"
        fi
        
        echo "Finished processing $domain"
        echo "=================================="
    done
    
    echo "All datasets processed!"
}

# Execute main function
main "$@"
