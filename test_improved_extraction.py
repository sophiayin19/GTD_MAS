from datasets.humaneval_dataset import humaneval_get_predict, check_correctness

# Test improved extraction
test_responses = [
    '''Here's the solution:

```python
def add(x, y):
    return x + y
```

This function adds two numbers.''',
    
    '''def multiply(a, b):
    return a * b

This multiplies a and b.''',
    
    '''The code is:
def subtract(x, y):
return x - y
This function subtracts y from x.'''
]

print('Testing improved code extraction:')
for i, test in enumerate(test_responses, 1):
    result = humaneval_get_predict(test)
    print(f'Test {i}: {repr(result)}')

print('\nTesting binary scoring:')
# Test binary scoring
prompt = 'def add(x, y): return x + y'
completion = 'def add(x, y): return x + y'
test = '''def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0

def test_check():
    check(add)

test_check()
'''

result, msg = check_correctness(prompt, completion, test)
print(f'Binary result: {result} (type: {type(result)}), Message: {msg}')
