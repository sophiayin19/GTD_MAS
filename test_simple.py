import sys
sys.path.append('.')

from datasets.humaneval_dataset import check_correctness

# Test cases
test_cases = [
    ("Perfect code", "def add(x, y): return x + y", "def add(x, y): return x + y"),
    ("Wrong code", "def add(x, y): return x + y", "def add(x, y): return x * y"),
    ("Syntax error", "def add(x, y): return x + y", "def add(x, y): return x +")
]

test = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0

def test_check():
    check(add)

test_check()
"""

for name, prompt, completion in test_cases:
    score, msg = check_correctness(prompt, completion, test)
    print(f"{name}: Score: {score}, Message: {msg}")
