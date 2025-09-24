#!/bin/bash

# Test the improved prompts with a small sample
echo "Testing improved prompts with 5 problems..."

# Run Phase 3 with improved prompts on a small sample
python experiments/run_humaneval.py \
    --mode GTD \
    --llm_name gpt-4o-mini \
    --domain humaneval \
    --agent_names "Programming_Expert" \
    --agent_nums 1 \
    --dataset_json datasets/humaneval/humaneval-py.jsonl \
    --num_problems 5 \
    --output_dir result/test_improved_prompts

echo "Test completed. Check result/test_improved_prompts/ for results."
