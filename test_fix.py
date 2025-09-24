import sys
sys.path.append('.')

from datasets.humaneval_dataset import check_correctness

# Test the partially correct case that was failing
prompt = "def add(x, y): return x + y"
completion = "def add(x, y): return x + y if x != 1 else 999"  # This should pass 2/3 tests
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
print(f"Fixed test result: Score: {score}, Message: {msg}")
