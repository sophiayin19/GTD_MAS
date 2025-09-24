import json
import sys
import re
import ast
sys.path.append('.')

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
        # This is tricky - we need to run tests individually
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

def test_partial_credit():
    """Test partial credit scoring on recent results."""
    
    # Load the latest results
    latest_file = 'result/gtd_humaneval/gpt-4o-mini_2025-09-07-22-33-09.json'
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print("=== TESTING PARTIAL CREDIT SCORING ===")
    
    total_binary_score = 0
    total_partial_score = 0
    
    for i, result in enumerate(results):
        question = result.get('Question', '')
        attempt_code = result.get('Attempt_Code', '')
        test = result.get('Test', '')
        
        # Binary scoring (current)
        from datasets.humaneval_dataset import check_correctness
        binary_score, binary_msg = check_correctness(question, attempt_code, test)
        binary_value = 1.0 if binary_score else 0.0
        
        # Partial credit scoring
        partial_value, partial_msg = check_correctness_partial(question, attempt_code, test)
        
        print(f"Problem {i+1}:")
        print(f"  Binary score: {binary_value}")
        print(f"  Partial score: {partial_value:.3f}")
        print(f"  Partial message: {partial_msg}")
        print()
        
        total_binary_score += binary_value
        total_partial_score += partial_value
    
    print(f"=== SUMMARY ===")
    print(f"Total problems: {len(results)}")
    print(f"Binary success rate: {total_binary_score/len(results):.3f}")
    print(f"Partial credit rate: {total_partial_score/len(results):.3f}")
    print(f"Improvement: {total_partial_score/len(results) - total_binary_score/len(results):.3f}")

if __name__ == "__main__":
    test_partial_credit()
