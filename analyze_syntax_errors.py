import json
import sys
sys.path.append('.')

from datasets.humaneval_dataset import humaneval_get_predict

def analyze_syntax_errors():
    """Analyze syntax errors in detail to understand what's going wrong."""
    
    latest_file = 'result/gtd_humaneval/gpt-4o-mini_2025-09-07-19-18-07.json'
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print("=== SYNTAX ERROR ANALYSIS ===")
    
    syntax_error_count = 0
    logic_error_count = 0
    success_count = 0
    
    for i, result in enumerate(results):
        solved = result.get('Solved', False)
        result_str = result.get('Result_Str', '')
        response = result.get('Response', '')
        
        if not solved:
            if 'SyntaxError' in result_str:
                syntax_error_count += 1
                if syntax_error_count <= 5:  # Show first 5 syntax errors
                    print(f"\n--- Syntax Error {syntax_error_count} ---")
                    print(f"Problem {i+1}")
                    
                    # Show the function name
                    question = result.get('Question', '')
                    if 'def ' in question:
                        func_name = question.split('def ')[1].split('(')[0]
                        print(f"Function: {func_name}")
                    
                    # Show the extracted code
                    extracted_code = humaneval_get_predict(response)
                    print(f"Extracted code:")
                    print(extracted_code)
                    print(f"Error: {result_str}")
                    print("-" * 50)
            else:
                logic_error_count += 1
        else:
            success_count += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Total problems: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Syntax errors: {syntax_error_count}")
    print(f"Logic errors: {logic_error_count}")
    print(f"Success rate: {success_count/len(results)*100:.1f}%")
    print(f"Syntax error rate: {syntax_error_count/len(results)*100:.1f}%")

if __name__ == "__main__":
    analyze_syntax_errors()
