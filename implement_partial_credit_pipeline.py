import json
import ast
import os

def count_test_cases(test_code):
    """Count the number of test cases (assertions) in the test code."""
    try:
        # Parse the test code to find assert statements
        tree = ast.parse(test_code)
        assert_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assert_count += 1
        
        return assert_count
    except:
        # Fallback: count 'assert' keywords
        return test_code.count('assert')

def check_correctness_partial(prompt, completion, test):
    """Check correctness with partial credit scoring."""
    program = f"{prompt}\n{completion}\n{test}"
    
    # Count total test cases
    total_tests = count_test_cases(test)
    
    if total_tests == 0:
        return 0.0, "No test cases found"
    
    try:
        # Execute the program
        exec_globals = {}
        exec(program, exec_globals)
        # If we get here, all tests passed
        return 1.0, f"All {total_tests} tests passed"
    except AssertionError as e:
        # Some tests failed, but we need to count how many passed
        passed_tests = 0
        
        # Extract individual test cases and run them
        test_lines = test.split('\n')
        for line in test_lines:
            if 'assert' in line and line.strip():
                try:
                    # Try to run just this assertion
                    single_test_program = f"{prompt}\n{completion}\n{line.strip()}"
                    exec(single_test_program, exec_globals)
                    passed_tests += 1
                except:
                    # This test failed
                    pass
        
        partial_score = passed_tests / total_tests
        return partial_score, f"{passed_tests}/{total_tests} tests passed"
    except Exception as e:
        return 0.0, f"Execution failed: {type(e).__name__}: {e}"

def update_humaneval_dataset():
    """Update the HumanEval dataset to use partial credit scoring."""
    
    dataset_file = 'datasets/humaneval_dataset.py'
    
    # Read the current file
    with open(dataset_file, 'r') as f:
        content = f.read()
    
    # Create the new check_correctness function with partial credit
    new_check_correctness = '''def check_correctness(prompt: str, completion: str, test: str) -> Tuple[float, str]:
    """
    Evaluates the generated code against the provided test cases.
    Returns a float score (0.0 to 1.0) representing the percentage of test cases passed.
    """
    program = f"{prompt}\\n{completion}\\n{test}"
    
    # Count total test cases
    total_tests = count_test_cases(test)
    
    if total_tests == 0:
        return 0.0, "No test cases found"
    
    try:
        # Execute the program
        exec_globals = {}
        exec(program, exec_globals)
        # If we get here, all tests passed
        return 1.0, f"All {total_tests} tests passed"
    except AssertionError as e:
        # Some tests failed, but we need to count how many passed
        passed_tests = 0
        
        # Extract individual test cases and run them
        test_lines = test.split('\\n')
        for line in test_lines:
            if 'assert' in line and line.strip():
                try:
                    # Try to run just this assertion
                    single_test_program = f"{prompt}\\n{completion}\\n{line.strip()}"
                    exec(single_test_program, exec_globals)
                    passed_tests += 1
                except:
                    # This test failed
                    pass
        
        partial_score = passed_tests / total_tests
        return partial_score, f"{passed_tests}/{total_tests} tests passed"
    except Exception as e:
        return 0.0, f"Execution failed: {type(e).__name__}: {e}"'''
    
    # Add the count_test_cases function
    count_test_cases_func = '''def count_test_cases(test_code):
    """Count the number of test cases (assertions) in the test code."""
    try:
        # Parse the test code to find assert statements
        tree = ast.parse(test_code)
        assert_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assert_count += 1
        
        return assert_count
    except:
        # Fallback: count 'assert' keywords
        return test_code.count('assert')'''
    
    # Find and replace the old check_correctness function
    import re
    
    # Pattern to match the old function
    old_pattern = r'def check_correctness\(prompt: str, completion: str, test: str\) -> Tuple\[bool, str\]:.*?(?=\n\ndef|\n\nclass|\Z)'
    
    # Replace with new function
    new_content = re.sub(old_pattern, new_check_correctness, content, flags=re.DOTALL)
    
    # Add the count_test_cases function after imports
    if 'def count_test_cases' not in new_content:
        # Find the imports section and add the function after it
        import_section = new_content.find('from typing import')
        if import_section != -1:
            # Find the end of the imports
            end_imports = new_content.find('\n\n', import_section)
            if end_imports != -1:
                new_content = new_content[:end_imports] + '\n\n' + count_test_cases_func + new_content[end_imports:]
    
    # Write the updated content back
    with open(dataset_file, 'w') as f:
        f.write(new_content)
    
    print("✓ Updated HumanEval dataset to use partial credit scoring")
    print("✓ Modified check_correctness function to return float scores (0.0 to 1.0)")
    print("✓ Added count_test_cases function to count individual test cases")

if __name__ == "__main__":
    update_humaneval_dataset()
