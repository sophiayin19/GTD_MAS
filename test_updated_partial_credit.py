import sys
sys.path.append('.')

from datasets.humaneval_dataset import check_correctness

def test_partial_credit():
    """Test the updated partial credit scoring."""
    
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
        # Wrong logic (should pass 1/3 tests)
        ("Wrong logic", "def add(x, y): return x - y"),
        # Syntax error
        ("Syntax error", "def add(x, y): return x +"),
    ]
    
    print("=== TESTING UPDATED PARTIAL CREDIT SCORING ===")
    
    for name, completion in test_cases:
        score, msg = check_correctness(prompt, completion, test)
        print(f"{name}: Score = {score:.3f}, Message = {msg}")

if __name__ == "__main__":
    test_partial_credit()
