import sys
import os
sys.path.append('.')

from datasets.humaneval_dataset import check_correctness, humaneval_get_predict
import json

# Test with some sample problems to see what's happening
def test_partial_credit():
    print("=== Testing Partial Credit Scoring ===\n")
    
    # Test 1: Perfect code
    print("Test 1: Perfect code")
    prompt = "def add(x, y): return x + y"
    completion = "def add(x, y): return x + y"
    test = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0

def test_check():
    check(add)

test_check()
"""
    score, msg = check_correctness(prompt, completion, test)
    print(f"Score: {score}, Message: {msg}")
    
    # Test 2: Partially correct code
    print("\nTest 2: Partially correct code (2/3 tests should pass)")
    prompt = "def add(x, y): return x + y"
    completion = "def add(x, y): return x + y if x != 1 else 999"  # This will fail the first test
    test = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0

def test_check():
    check(add)

test_check()
"""
    score, msg = check_correctness(prompt, completion, test)
    print(f"Score: {score}, Message: {msg}")
    
    # Test 3: Completely wrong code
    print("\nTest 3: Completely wrong code")
    prompt = "def add(x, y): return x + y"
    completion = "def add(x, y): return x * y"  # This will fail all tests
    test = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0

def test_check():
    check(add)

test_check()
"""
    score, msg = check_correctness(prompt, completion, test)
    print(f"Score: {score}, Message: {msg}")
    
    # Test 4: Syntax error
    print("\nTest 4: Syntax error")
    prompt = "def add(x, y): return x + y"
    completion = "def add(x, y): return x +"  # Syntax error
    test = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0

def test_check():
    check(add)

test_check()
"""
    score, msg = check_correctness(prompt, completion, test)
    print(f"Score: {score}, Message: {msg}")

if __name__ == "__main__":
    test_partial_credit()
