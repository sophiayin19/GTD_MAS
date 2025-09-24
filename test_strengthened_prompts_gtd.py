import json
import sys
import os
sys.path.append(".")

def test_strengthened_prompts_gtd():
    """Test the strengthened prompts with a small GTD run."""
    
    # Create a small test dataset with just 3 problems
    with open("datasets/humaneval/humaneval-py.jsonl", "r") as f:
        problems = [json.loads(line) for line in f][:3]
    
    # Save as a small test dataset
    with open("datasets/humaneval/humaneval-test-small.jsonl", "w") as f:
        for problem in problems:
            f.write(json.dumps(problem) + "\\n")
    
    print("=== TESTING STRENGTHENED PROMPTS WITH GTD ===")
    print("Created small test dataset with 3 problems")
    print("Now run the GTD framework with strengthened prompts:")
    print()
    print("python experiments/run_humaneval.py \\\\")
    print("    --mode GTD \\\\")
    print("    --llm_name gpt-4o-mini \\\\")
    print("    --domain humaneval \\\\")
    print("    --agent_names Programming_Expert \\\\")
    print("    --agent_nums 1 \\\\")
    print("    --dataset_json datasets/humaneval/humaneval-test-small.jsonl \\\\")
    print("    --gtd-proxy-model-path trained_models/proxy_model_humaneval.pth \\\\")
    print("    --gtd-diffusion-model-path trained_models/diffusion_model_humaneval.pth")

if __name__ == "__main__":
    test_strengthened_prompts_gtd()
