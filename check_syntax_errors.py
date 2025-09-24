import json
import os

def check_syntax_errors():
    """Check for syntax errors and other issues in the latest results."""
    
    # Find the most recent result file
    result_files = []
    if os.path.exists('result/gtd_humaneval'):
        for file in os.listdir('result/gtd_humaneval'):
            if file.endswith('.json'):
                result_files.append(os.path.join('result/gtd_humaneval', file))
    
    if result_files:
        latest_file = max(result_files, key=os.path.getmtime)
        print(f"Analyzing: {latest_file}")
        
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        print(f"Total problems: {len(results)}")
        
        # Analyze the results
        syntax_errors = 0
        logic_errors = 0
        successful = 0
        other_errors = 0
        
        print("\n=== ERROR ANALYSIS ===")
        
        for i, result in enumerate(results):
            solved = result.get('Solved', False)
            result_str = result.get('Result_Str', '')
            attempt_code = result.get('Attempt_Code', '')
            
            if solved:
                successful += 1
            else:
                if 'SyntaxError' in result_str:
                    syntax_errors += 1
                    print(f"Problem {i+1}: SyntaxError - {result_str}")
                    print(f"  Code: {attempt_code[:100]}...")
                    print()
                elif 'Test failed' in result_str:
                    logic_errors += 1
                    print(f"Problem {i+1}: Logic error - {result_str}")
                    print(f"  Code: {attempt_code[:100]}...")
                    print()
                else:
                    other_errors += 1
                    print(f"Problem {i+1}: Other error - {result_str}")
                    print(f"  Code: {attempt_code[:100]}...")
                    print()
        
        print(f"\n=== SUMMARY ===")
        print(f"Successful: {successful}")
        print(f"Syntax errors: {syntax_errors}")
        print(f"Logic errors: {logic_errors}")
        print(f"Other errors: {other_errors}")
        print(f"Success rate: {successful/len(results)*100:.1f}%")
        print(f"Syntax error rate: {syntax_errors/len(results)*100:.1f}%")
        
        if syntax_errors > 0:
            print(f"\n⚠️  Still have {syntax_errors} syntax errors despite improved prompts")
        else:
            print(f"\n✅ No syntax errors found - improved prompts are working!")
            
    else:
        print("No result files found")

if __name__ == "__main__":
    check_syntax_errors()
