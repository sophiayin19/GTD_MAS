import json
import sys
import re
import ast
import os
sys.path.append(".")

def count_test_cases(test_code):
    """Count the number of test cases (assertions) in the test code."""
    try:
        tree = ast.parse(test_code)
        assert_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assert_count += 1
        return assert_count
    except:
        return test_code.count('assert')

def check_correctness_partial(prompt, completion, test):
    """Check correctness with partial credit scoring."""
    program = f"{prompt}\n{completion}\n{test}"
    total_tests = count_test_cases(test)
    
    if total_tests == 0:
        return 0.0, "No test cases found"
    
    try:
        exec_globals = {}
        exec(program, exec_globals)
        return 1.0, f"All {total_tests} tests passed"
    except AssertionError as e:
        passed_tests = 0
        test_lines = test.split('\n')
        for line in test_lines:
            if 'assert' in line and line.strip():
                try:
                    single_test_program = f"{prompt}\n{completion}\n{line.strip()}"
                    exec(single_test_program, exec_globals)
                    passed_tests += 1
                except:
                    pass
        
        partial_score = passed_tests / total_tests
        return partial_score, f"{passed_tests}/{total_tests} tests passed"
    except Exception as e:
        return 0.0, f"Execution failed: {type(e).__name__}: {e}"

def test_partial_credit_full():
    """Test partial credit scoring on the full results."""
    
