import sys
import os
sys.path.append('.')

from datasets.humaneval_dataset import humaneval_get_predict, check_correctness
import json

def analyze_model_responses():
    print("=== Analyzing Model Responses ===\n")
    
    # Load a few sample problems from the dataset
    try:
        with open('datasets/humaneval/humaneval-py.jsonl', 'r') as f:
            problems = [json.loads(line) for line in f.readlines()[:5]]  # First 5 problems
    except:
        print("Could not load humaneval dataset. Let's use sample problems instead.")
        problems = [
            {
                "prompt": "def add(x, y):\n    \"\"\"Add two numbers x and y\"\"\"\n    ",
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n    assert candidate(-1, 1) == 0\n\ndef test_check():\n    check(add)\n\ntest_check()\n"
            }
        ]
    
    print(f"Analyzing {len(problems)} problems...\n")
    
    for i, problem in enumerate(problems):
        print(f"Problem {i+1}:")
        print(f"Prompt: {problem['prompt'][:100]}...")
        print(f"Test: {problem['test'][:100]}...")
        
        # Simulate some model responses to see what might be happening
        sample_responses = [
            "def add(x, y): return x + y",  # Perfect
            "```python\ndef add(x, y): return x + y\n```",  # In markdown
            "Here's the solution:\n```python\ndef add(x, y): return x + y\n```",  # With text
            "def add(x, y): return x * y",  # Wrong logic
            "def add(x, y):\n    # Add two numbers\n    return x + y",  # With comments
            "def add(x, y):\n    result = x + y\n    return result",  # Verbose
        ]
        
        print("Testing sample responses:")
        for j, response in enumerate(sample_responses):
            extracted = humaneval_get_predict(response)
            score, msg = check_correctness(problem['prompt'], extracted, problem['test'])
            print(f"  Response {j+1}: Score: {score}, Message: {msg}")
            print(f"    Extracted: {repr(extracted[:50])}...")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    analyze_model_responses()
