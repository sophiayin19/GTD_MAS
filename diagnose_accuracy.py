import json
import sys
sys.path.append('.')

from datasets.humaneval_dataset import humaneval_get_predict, check_correctness

def analyze_responses():
    """Analyze actual model responses to understand why accuracy is low."""
    
    # Sample some problems from the dataset
    with open('datasets/humaneval/humaneval-py.jsonl', 'r') as f:
        problems = [json.loads(line) for line in f][:5]  # First 5 problems
    
    print("=== DIAGNOSTIC ANALYSIS ===")
    print("Analyzing why accuracy is low...")
    print()
    
    for i, problem in enumerate(problems, 1):
        print(f"Problem {i}: {problem['name']}")
        print(f"Function: {problem['prompt'].split('def ')[1].split('(')[0] if 'def ' in problem['prompt'] else 'Unknown'}")
        
        # Extract the function signature
        prompt_lines = problem['prompt'].split('\n')
        func_sig = None
        for line in prompt_lines:
            if line.strip().startswith('def '):
                func_sig = line.strip()
                break
        
        print(f"Signature: {func_sig}")
        print(f"Test cases: {problem['test'].count('assert')} assertions")
        print()
        
        # Show what a perfect response should look like
        print("What a perfect response should look like:")
        if 'canonical_solution' in problem:
            print(f"Canonical: {problem['canonical_solution'][:100]}...")
        else:
            print("No canonical solution available")
        print("-" * 50)
        print()

def test_code_extraction():
    """Test if code extraction is working properly."""
    print("=== CODE EXTRACTION TEST ===")
    
    test_responses = [
        # Perfect response
        '''```python
def add(x, y):
    return x + y
```''',
        
        # Response with explanation
        '''Here's the solution:

```python
def multiply(a, b):
    return a * b
```

This function multiplies two numbers.''',
        
        # Malformed response
        '''The function should add two numbers together.
def add(x, y):
    return x + y
This is the correct implementation.''',
        
        # No code block
        '''def subtract(x, y):
    return x - y'''
    ]
    
    for i, response in enumerate(test_responses, 1):
        extracted = humaneval_get_predict(response)
        print(f"Test {i}:")
        print(f"Original: {repr(response[:50])}...")
        print(f"Extracted: {repr(extracted)}")
        print(f"Valid function: {'def ' in extracted and '(' in extracted}")
        print()

def test_scoring():
    """Test if scoring is working properly."""
    print("=== SCORING TEST ===")
    
    # Test with a simple function
    prompt = "def add(x, y): return x + y"
    test = '''def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0

def test_check():
    check(add)

test_check()
'''
    
    test_cases = [
        # Perfect code
        ("Perfect", "def add(x, y): return x + y"),
        # Wrong logic
        ("Wrong logic", "def add(x, y): return x - y"),
        # Syntax error
        ("Syntax error", "def add(x, y): return x +"),
        # Missing function
        ("Missing function", "print('hello')"),
    ]
    
    for name, completion in test_cases:
        score, msg = check_correctness(prompt, completion, test)
        print(f"{name}: Score = {score}, Message = {msg}")

if __name__ == "__main__":
    analyze_responses()
    test_code_extraction()
    test_scoring()
