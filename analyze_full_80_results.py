import json
import os
import sys
sys.path.append('.')

from datasets.humaneval_dataset import check_correctness

def analyze_full_80_results():
    """Analyze the results from the full 80-problem test to categorize failures."""
    
    # Find the most recent result file
    result_files = []
    if os.path.exists('result/gtd_humaneval'):
        for file in os.listdir('result/gtd_humaneval'):
            if file.endswith('.json'):
                result_files.append(os.path.join('result/gtd_humaneval', file))
    
    if result_files:
        latest_file = max(result_files, key=os.path.getmtime)
        print(f"Analyzing: {latest_file}")
    else:
        print("No result files found!")
        return
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print("=== ANALYSIS OF FULL 80-PROBLEM TEST RESULTS ===")
    print(f"Total problems: {len(results)}")
    print()
    
    # Categorize the results
    successful = 0
    syntax_errors = 0
    logic_errors = 0
    import_errors = 0
    other_errors = 0
    
    print("Detailed Analysis (first 10 problems):")
    print("-" * 50)
    
    for i, result in enumerate(results[:10], 1):  # Show first 10 for detail
        problem_name = result.get('name', f'Problem {i}')
        score = result.get('score', 0.0)
        error_msg = result.get('error_message', '')
        
        print(f"Problem {i}: {problem_name}")
        print(f"  Score: {score}")
        
        if score == 1.0:
            print(f"  Status: ✅ SUCCESS")
            successful += 1
        elif score == 0.0:
            if 'SyntaxError' in error_msg or 'invalid syntax' in error_msg:
                print(f"  Status: ❌ SYNTAX ERROR")
                print(f"  Error: {error_msg}")
                syntax_errors += 1
            elif 'NameError' in error_msg or 'is not defined' in error_msg:
                print(f"  Status: ❌ IMPORT ERROR")
                print(f"  Error: {error_msg}")
                import_errors += 1
            elif 'Test failed' in error_msg or 'AssertionError' in error_msg:
                print(f"  Status: ❌ LOGIC ERROR")
                print(f"  Error: {error_msg}")
                logic_errors += 1
            else:
                print(f"  Status: ❌ OTHER ERROR")
                print(f"  Error: {error_msg}")
                other_errors += 1
        else:
            print(f"  Status: ⚠️ PARTIAL SUCCESS")
            print(f"  Error: {error_msg}")
            logic_errors += 1
        
        print()
    
    # Count all problems for summary
    for result in results:
        score = result.get('score', 0.0)
        error_msg = result.get('error_message', '')
        
        if score == 1.0:
            successful += 1
        elif score == 0.0:
            if 'SyntaxError' in error_msg or 'invalid syntax' in error_msg:
                syntax_errors += 1
            elif 'NameError' in error_msg or 'is not defined' in error_msg:
                import_errors += 1
            elif 'Test failed' in error_msg or 'AssertionError' in error_msg:
                logic_errors += 1
            else:
                other_errors += 1
        else:
            logic_errors += 1
    
    # Summary
    print("=== SUMMARY (ALL 80 PROBLEMS) ===")
    print(f"✅ Successful: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"❌ Syntax Errors: {syntax_errors}/{len(results)} ({syntax_errors/len(results)*100:.1f}%)")
    print(f"❌ Logic Errors: {logic_errors}/{len(results)} ({logic_errors/len(results)*100:.1f}%)")
    print(f"❌ Import Errors: {import_errors}/{len(results)} ({import_errors/len(results)*100:.1f}%)")
    print(f"❌ Other Errors: {other_errors}/{len(results)} ({other_errors/len(results)*100:.1f}%)")
    print()
    
    # Improvement analysis
    print("=== IMPROVEMENT ANALYSIS ===")
    print(f"Success Rate: {successful/len(results)*100:.1f}%")
    print(f"Syntax Error Rate: {syntax_errors/len(results)*100:.1f}%")
    print(f"Logic Error Rate: {logic_errors/len(results)*100:.1f}%")
    print()
    
    # Compare with previous results
    print("=== COMPARISON WITH PREVIOUS RESULTS ===")
    print("Previous results (before strengthened prompts):")
    print("- Success Rate: ~66.2%")
    print("- Syntax Error Rate: ~17.5%")
    print("- Logic Error Rate: ~16.3%")
    print()
    print("Current results (with strengthened prompts):")
    print(f"- Success Rate: {successful/len(results)*100:.1f}%")
    print(f"- Syntax Error Rate: {syntax_errors/len(results)*100:.1f}%")
    print(f"- Logic Error Rate: {logic_errors/len(results)*100:.1f}%")
    print()
    
    # Calculate improvement
    success_improvement = (successful/len(results)*100) - 66.2
    syntax_improvement = 17.5 - (syntax_errors/len(results)*100)
    
    print("=== IMPROVEMENT METRICS ===")
    print(f"Success Rate Change: {success_improvement:+.1f} percentage points")
    print(f"Syntax Error Reduction: {syntax_improvement:+.1f} percentage points")
    
    if syntax_errors == 0:
        print("🎉 EXCELLENT: No syntax errors! The strengthened prompts completely eliminated syntax errors!")
    elif syntax_errors < len(results) * 0.1:
        print("✅ EXCELLENT: Very low syntax error rate. Strengthened prompts are working very well!")
    elif syntax_errors < len(results) * 0.2:
        print("✅ GOOD: Low syntax error rate. Strengthened prompts are helping!")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Still have syntax errors. May need further prompt strengthening.")

if __name__ == "__main__":
    analyze_full_80_results()
