import sys
sys.path.append('.')

from datasets.humaneval_dataset import check_correctness, humaneval_get_predict

# Test cases with various response formats
test_cases = [
    ("Perfect code", "def add(x, y): return x + y"),
    ("Code in markdown", "```python\ndef add(x, y): return x + y\n```"),
    ("Code with extra text", "Here's the solution:\n```python\ndef add(x, y): return x + y\n```\nThis should work."),
    ("Code with indentation issues", "def add(x, y):\nreturn x + y"),
    ("Wrong code", "def add(x, y): return x * y")
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

for name, completion in test_cases:
    score, msg = check_correctness("def add(x, y): return x + y", completion, test)
    print(f"{name}: Score: {score}, Message: {msg}")
