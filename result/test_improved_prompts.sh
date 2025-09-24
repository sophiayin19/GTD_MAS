#!/bin/bash

# Test the improved prompts with a small sample
echo "Testing improved prompts with 5 problems..."

# Create a small test dataset with just 5 problems
head -5 datasets/humaneval/humaneval-py.jsonl > datasets/humaneval/humaneval-test.jsonl

# Run Phase 3 with improved prompts on the small test dataset
python experiments/run_humaneval.py \
    --mode GTD \
    --llm_name gpt-4o-mini \
    --domain humaneval \
    --agent_names "Programming_Expert" \
    --agent_nums 1 \
    --dataset_json datasets/humaneval/humaneval-test.jsonl

echo "Test completed. Check the result directory for the latest results."
