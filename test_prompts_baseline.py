import json
import sys
sys.path.append('.')

from datasets.humaneval_dataset import humaneval_get_predict, check_correctness
from GDesigner.prompt.humaneval_prompt_set import HumanEvalPromptSet

def test_improved_prompts():
    """Test the improved prompts by simulating agent responses."""
    
    # Load a few test problems
    with open('datasets/humaneval/humaneval-py.jsonl', 'r') as f:
        problems = [json.loads(line) for line in f][:3]
    
    print("=== TESTING IMPROVED PROMPTS ===")
    
    # Get the improved prompt for Programming Expert
    prompt_set = HumanEvalPromptSet()
    programming_expert_prompt = prompt_set.get_constraint("Programming Expert")
    
    print("Programming Expert prompt:")
    print(programming_expert_prompt[:200] + "...")
    print()
    
    # Check if CRITICAL requirements are in the prompt
    if "CRITICAL" in programming_expert_prompt:
        print("✓ CRITICAL requirements found in prompt")
    else:
        print("✗ CRITICAL requirements NOT found in prompt")
    
    # Test with a simple problem
    problem = problems[0]
    print(f"\nTesting with problem: {problem['name']}")
    print(f"Function: {problem['prompt'].split('def ')[1].split('(')[0] if 'def ' in problem['prompt'] else 'Unknown'}")
    
    # Show what the prompt looks like
    print(f"\nTask prompt: {problem['prompt'][:100]}...")
    
    print("\n✓ Prompt testing completed")

if __name__ == "__main__":
    test_improved_prompts()
