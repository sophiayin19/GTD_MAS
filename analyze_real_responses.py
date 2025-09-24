import json
import sys
import os
sys.path.append('.')

from datasets.humaneval_dataset import humaneval_get_predict, check_correctness

def analyze_recent_results():
    """Analyze the actual results from your recent runs."""
    
    # Check the gtd_humaneval directory
    humaneval_dir = 'result/gtd_humaneval'
    
    if not os.path.exists(humaneval_dir):
        print("No gtd_humaneval directory found.")
        return
    
    print(f"Files in {humaneval_dir}:")
    files = os.listdir(humaneval_dir)
    for file in files:
        print(f"  - {file}")
    
    # Look for JSON result files
    json_files = [f for f in files if f.endswith('.json')]
    
    if not json_files:
        print("No JSON result files found.")
        return
    
    # Analyze the most recent result file
    latest_file = max([os.path.join(humaneval_dir, f) for f in json_files], 
                     key=os.path.getmtime)
    print(f"\nAnalyzing latest result file: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print(f"Total problems: {len(results)}")
    
    # Analyze the results
    total_score = 0
    syntax_errors = 0
    logic_errors = 0
    successful = 0
    
    for i, result in enumerate(results[:10]):  # Analyze first 10
        print(f"\n--- Problem {i+1} ---")
        
        if 'task' in result:
            task = result['task']
            print(f"Task: {task[:100]}...")
        
        if 'model_response' in result:
            response = result['model_response']
            extracted_code = humaneval_get_predict(response)
            
            print(f"Raw response length: {len(response)} characters")
            print(f"Extracted code length: {len(extracted_code)} characters")
            print(f"Extracted code preview: {extracted_code[:100]}...")
            
            # Check if it's a valid function
            is_valid_function = 'def ' in extracted_code and '(' in extracted_code
            print(f"Valid function: {is_valid_function}")
            
            if 'score' in result:
                score = result['score']
                total_score += score
                print(f"Score: {score}")
                
                if score == 0:
                    if 'error' in result:
                        error = result['error']
                        if 'SyntaxError' in error:
                            syntax_errors += 1
                            print(f"Syntax error: {error}")
                        else:
                            logic_errors += 1
                            print(f"Logic error: {error}")
                    else:
                        logic_errors += 1
                        print("Logic error (no specific error message)")
                else:
                    successful += 1
                    print("Success!")
        else:
            print("No model response found")
    
    print(f"\n=== SUMMARY (first 10 problems) ===")
    print(f"Successful: {successful}/10")
    print(f"Syntax errors: {syntax_errors}/10")
    print(f"Logic errors: {logic_errors}/10")
    print(f"Average score: {total_score/10:.2f}")

if __name__ == "__main__":
    analyze_recent_results()
