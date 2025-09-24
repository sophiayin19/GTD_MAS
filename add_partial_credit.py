import re

def add_partial_credit():
    """Add partial credit scoring to the HumanEval dataset."""
    
    # Read the current file
    with open('datasets/humaneval_dataset.py', 'r') as f:
        content = f.read()
    
    # Add the count_test_cases function after imports
    count_test_cases_func = '''
def count_test_cases(test_code):
    """Count the number of test cases (assertions) in the test code."""
    try:
        import ast
        tree = ast.parse(test_code)
        assert_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assert_count += 1
        return assert_count
    except:
        return test_code.count('assert')
'''
    
    # Add the function after the imports
    if 'def count_test_cases' not in content:
        # Find the end of imports
        end_imports = content.find('from typing import')
        if end_imports != -1:
            end_imports = content.find('\n\n', end_imports)
            if end_imports != -1:
                content = content[:end_imports] + count_test_cases_func + content[end_imports:]
    
    # Update the check_correctness function signature and return type
    content = content.replace(
        'def check_correctness(prompt: str, completion: str, test: str) -> Tuple[bool, str]:',
        'def check_correctness(prompt: str, completion: str, test: str) -> Tuple[float, str]:'
    )
    
    # Update the function body to use partial credit
    old_body = '''    """
    Evaluates the generated code against the provided test cases.
    Returns True if all tests pass, False otherwise.
    """
    program = f"{prompt}\\n{completion}\\n{test}"
    try:
        exec_globals = {}
        exec(program, exec_globals)
        return True, "All tests passed"
    except AssertionError as e:
        return False, f"Test failed: {e}"
    except Exception as e:
        return False, f"Execution failed: {type(e).__name__}: {e}"'''
    
    new_body = '''    """
    Evaluates the generated code against the provided test cases.
    Returns a float score (0.0 to 1.0) representing the percentage of test cases passed.
    """
    program = f"{prompt}\\n{completion}\\n{test}"
    
    # Count total test cases
    total_tests = count_test_cases(test)
    
    if total_tests == 0:
        return 0.0, "No test cases found"
    
    try:
        exec_globals = {}
        exec(program, exec_globals)
        return 1.0, f"All {total_tests} tests passed"
    except AssertionError as e:
        # Some tests failed, count how many passed
        passed_tests = 0
        test_lines = test.split('\\n')
        for line in test_lines:
            if 'assert' in line and line.strip():
                try:
                    single_test_program = f"{prompt}\\n{completion}\\n{line.strip()}"
                    exec(single_test_program, exec_globals)
                    passed_tests += 1
                except:
                    pass
        
        partial_score = passed_tests / total_tests
        return partial_score, f"{passed_tests}/{total_tests} tests passed"
    except Exception as e:
        return 0.0, f"Execution failed: {type(e).__name__}: {e}"'''
    
    # Replace the function body
    content = content.replace(old_body, new_body)
    
    # Write the updated content back
    with open('datasets/humaneval_dataset.py', 'w') as f:
        f.write(content)
    
    print("✓ Added partial credit scoring to HumanEval dataset")
    print("✓ Updated check_correctness function to return float scores")
    print("✓ Added count_test_cases function")

if __name__ == "__main__":
    add_partial_credit()
