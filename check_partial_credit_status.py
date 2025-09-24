import sys
sys.path.append('.')

from datasets.humaneval_dataset import check_correctness

def test_partial_credit_status():
    """Check if partial credit scoring is actually working."""
    
    # Test with a simple function that should get partial credit
    prompt = "def add(x, y): return x + y"
    test = '''def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0

def test_check():
    check(add)

test_check()
'''
    
    # Test with wrong logic (should get partial credit)
    wrong_logic = "def add(x, y): return x - y"
    
    print("=== CHECKING PARTIAL CREDIT STATUS ===")
    
    # Check the function signature
    import inspect
    sig = inspect.signature(check_correctness)
    print(f"Function signature: {sig}")
    
    # Test the function
    score, msg = check_correctness(prompt, wrong_logic, test)
    print(f"Wrong logic test: Score = {score}, Message = {msg}")
    
    # Check if score is float (partial credit) or bool (binary)
    print(f"Score type: {type(score)}")
    print(f"Score value: {score}")
    
    if isinstance(score, float):
        print("✓ Partial credit scoring is working!")
    else:
        print("✗ Still using binary scoring")

if __name__ == "__main__":
    test_partial_credit_status()
