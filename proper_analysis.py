import json
import sys
sys.path.append('.')

from datasets.humaneval_dataset import humaneval_get_predict, check_correctness

def analyze_humaneval_results():
    """Analyze the actual HumanEval results with correct key names."""
    
    latest_file = 'result/gtd_humaneval/gpt-4o-mini_2025-09-07-19-18-07.json'
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print(f"Total problems: {len(results)}")
    
    # Analyze the results
    successful = 0
    failed = 0
    syntax_errors = 0
    logic_errors = 0
    
    print("\n=== DETAILED ANALYSIS ===")
    
    for i, result in enumerate(results[:10]):  # Analyze first 10
        print(f"\n--- Problem {i+1} ---")
        
        # Extract key information
        question = result.get('Question', '')
        response = result.get('Response', '')
        attempt_code = result.get('Attempt_Code', '')
        solved = result.get('Solved', False)
        result_str = result.get('Result_Str', '')
        
        # Show function name
        if 'def ' in question:
            func_name = question.split('def ')[1].split('(')[0]
            print(f"Function: {func_name}")
        
        print(f"Solved: {solved}")
        print(f"Result: {result_str}")
        
        # Analyze the response
        if response:
            extracted_code = humaneval_get_predict(response)
            print(f"Response length: {len(response)} chars")
            print(f"Extracted code length: {len(extracted_code)} chars")
            
            # Check if extracted code matches attempt code
            if extracted_code.strip() == attempt_code.strip():
                print("✓ Code extraction successful")
            else:
                print("✗ Code extraction mismatch")
                print(f"Extracted: {extracted_code[:100]}...")
                print(f"Attempt: {attempt_code[:100]}...")
            
            # Check if it's a valid function
            is_valid = 'def ' in extracted_code and '(' in extracted_code
            print(f"Valid function: {is_valid}")
            
            if solved:
                successful += 1
                print("✓ SUCCESS")
            else:
                failed += 1
                if 'SyntaxError' in result_str:
                    syntax_errors += 1
                    print("✗ SYNTAX ERROR")
                else:
                    logic_errors += 1
                    print("✗ LOGIC ERROR")
        else:
            print("No response found")
            failed += 1
    
    print(f"\n=== SUMMARY (first 10 problems) ===")
    print(f"Successful: {successful}/10")
    print(f"Failed: {failed}/10")
    print(f"  - Syntax errors: {syntax_errors}")
    print(f"  - Logic errors: {logic_errors}")
    print(f"Success rate: {successful/10*100:.1f}%")
    
    # Overall summary
    total_successful = sum(1 for r in results if r.get('Solved', False))
    total_failed = len(results) - total_successful
    print(f"\n=== OVERALL SUMMARY (all {len(results)} problems) ===")
    print(f"Successful: {total_successful}/{len(results)}")
    print(f"Failed: {total_failed}/{len(results)}")
    print(f"Overall success rate: {total_successful/len(results)*100:.1f}%")

if __name__ == "__main__":
    analyze_humaneval_results()
