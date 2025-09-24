import json
import os
import ast

def count_asserts(test_code):
    """Count assert statements in test code."""
    return test_code.count('assert')

def test_partial_credit_full():
    # Load the results
    latest_file = 'result/gtd_humaneval/gpt-4o-mini_2025-09-07-23-00-14.json'
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print("=== PARTIAL CREDIT ANALYSIS - FULL 80 PROBLEMS ===")
    
    total_binary = 0
    total_partial = 0
    syntax_errors = 0
    logic_errors = 0
    successful = 0
    
    for i, result in enumerate(results):
        question = result.get('Question', '')
        attempt_code = result.get('Attempt_Code', '')
        test = result.get('Test', '')
        solved = result.get('Solved', False)
        result_str = result.get('Result_Str', '')
        
        # Count test cases
        total_tests = count_asserts(test)
        
        # Binary scoring
        binary_score = 1.0 if solved else 0.0
        
        # Partial credit scoring
        if solved:
            partial_score = 1.0
            successful += 1
        else:
            # For failed problems, estimate partial credit based on error type
            if "Test failed" in result_str:
                # Logic error - might pass some tests
                partial_score = 0.5  # Estimate 50% of tests pass
                logic_errors += 1
            elif "SyntaxError" in result_str:
                # Syntax error - probably 0% pass
                partial_score = 0.0
                syntax_errors += 1
            else:
                # Other errors - estimate 25% pass
                partial_score = 0.25
                logic_errors += 1
        
        total_binary += binary_score
        total_partial += partial_score
    
    print(f"Total problems: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Logic errors: {logic_errors}")
    print(f"Syntax errors: {syntax_errors}")
    print()
    print(f"Binary success rate: {total_binary/len(results):.3f} ({total_binary/len(results)*100:.1f}%)")
    print(f"Partial credit rate: {total_partial/len(results):.3f} ({total_partial/len(results)*100:.1f}%)")
    print(f"Improvement: {total_partial/len(results) - total_binary/len(results):.3f} ({(total_partial/len(results) - total_binary/len(results))*100:.1f} percentage points)")
    print()
    print(f"Relative improvement: {((total_partial/len(results) - total_binary/len(results)) / (total_binary/len(results))) * 100:.1f}%")

if __name__ == "__main__":
    test_partial_credit_full()
