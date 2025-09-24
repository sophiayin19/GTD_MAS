import json
import os
import ast

def count_asserts(test_code):
    """Count assert statements in test code."""
    return test_code.count('assert')

def test_partial_credit():
    # Load the results
    latest_file = 'result/gtd_humaneval/gpt-4o-mini_2025-09-07-23-00-14.json'
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print("=== PARTIAL CREDIT ANALYSIS ===")
    
    total_binary = 0
    total_partial = 0
    
    # Test first 10 problems
    for i, result in enumerate(results[:10]):
        question = result.get('Question', '')
        attempt_code = result.get('Attempt_Code', '')
        test = result.get('Test', '')
        solved = result.get('Solved', False)
        
        # Count test cases
        total_tests = count_asserts(test)
        
        # Binary scoring
        binary_score = 1.0 if solved else 0.0
        
        # Simulate partial credit for failed problems
        if solved:
            partial_score = 1.0
        else:
            # For failed problems, estimate partial credit
            # This is a simplified version - in reality we'd run each test individually
            if "Test failed" in result.get('Result_Str', ''):
                # Logic error - might pass some tests
                partial_score = 0.5  # Estimate 50% of tests pass
            else:
                # Syntax error - probably 0% pass
                partial_score = 0.0
        
        print(f"Problem {i+1}: Binary={binary_score}, Partial={partial_score:.1f}, Tests={total_tests}")
        
        total_binary += binary_score
        total_partial += partial_score
    
    print(f"\nSummary (first 10):")
    print(f"Binary rate: {total_binary/10:.3f}")
    print(f"Partial rate: {total_partial/10:.3f}")
    print(f"Improvement: {total_partial/10 - total_binary/10:.3f}")

if __name__ == "__main__":
    test_partial_credit()
