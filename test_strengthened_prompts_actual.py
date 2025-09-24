import json
import sys
import os
sys.path.append('.')

from datasets.humaneval_dataset import check_correctness

def test_strengthened_prompts():
    """Test the strengthened prompts on a small sample to see if they reduce syntax errors."""
    
    # Load a few test problems
    with open('datasets/humaneval/humaneval-py.jsonl', 'r') as f:
        problems = [json.loads(line) for line in f][:5]  # First 5 problems
    
    print("=== TESTING STRENGTHENED PROMPTS ON SAMPLE PROBLEMS ===")
    
    # Simulate what the strengthened prompts should generate
    # (In reality, this would come from the GTD framework with the new prompts)
    
    # Test with some sample responses that should be improved by strengthened prompts
    test_cases = [
        # This should now generate actual code instead of descriptions
        {
            "name": "Test 1 - Should generate code, not description",
            "prompt": "def add(x, y): return x + y",
            "completion": "def add(x, y): return x + y",  # Good code
            "test": "assert add(1, 2) == 3"
        },
        # This is what we want to avoid (descriptions)
        {
            "name": "Test 2 - Description (should be avoided)",
            "prompt": "def add(x, y): return x + y", 
            "completion": "Define a function that adds two numbers together",  # Bad - description
            "test": "assert add(1, 2) == 3"
        },
        # This is what we want (actual code)
        {
            "name": "Test 3 - Actual code (good)",
            "prompt": "def multiply(x, y): return x * y",
            "completion": "def multiply(x, y): return x * y",  # Good code
            "test": "assert multiply(2, 3) == 6"
        }
    ]
    
    print("Testing sample responses to see if strengthened prompts would help:")
    print()
    
    for test_case in test_cases:
        print(f"{test_case['name']}:")
        print(f"  Completion: {test_case['completion']}")
        
        # Test the completion
        score, msg = check_correctness(test_case['prompt'], test_case['completion'], test_case['test'])
        print(f"  Score: {score:.3f}")
        print(f"  Message: {msg}")
        print()
    
    print("=== SUMMARY ===")
    print("The strengthened prompts should prevent agents from generating descriptions")
    print("like 'Define a function...' and force them to generate actual Python code.")
    print("This should reduce syntax errors significantly.")

if __name__ == "__main__":
    test_strengthened_prompts()
