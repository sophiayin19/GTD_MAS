import json
import os
import sys
sys.path.append('.')

def analyze_new_pipeline_results():
    """Analyze the results from the Programming Expert + Test Analyst pipeline."""
    
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
    
    print("=== ANALYSIS OF PROGRAMMING EXPERT + TEST ANALYST PIPELINE ===")
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
        solved = result.get('Solved', 0.0)
        result_str = result.get('Result_Str', '')
        response = result.get('Response', '')
        
        print(f"Problem {i}:")
        print(f"  Solved: {solved}")
        print(f"  Result: {result_str}")
        
        if solved == 1.0:
            print(f"  Status: ✅ SUCCESS")
            successful += 1
        elif solved == 0.0:
            if 'SyntaxError' in result_str or 'invalid syntax' in result_str:
                print(f"  Status: ❌ SYNTAX ERROR")
                syntax_errors += 1
            elif 'NameError' in result_str or 'is not defined' in result_str:
                print(f"  Status: ❌ IMPORT ERROR")
                import_errors += 1
            elif 'Test failed' in result_str or 'AssertionError' in result_str:
                print(f"  Status: ❌ LOGIC ERROR")
                logic_errors += 1
            else:
                print(f"  Status: ❌ OTHER ERROR")
                other_errors += 1
        else:
            print(f"  Status: ⚠️ PARTIAL SUCCESS")
            logic_errors += 1
        
        print()
    
    # Count all problems for summary
    for result in results:
        solved = result.get('Solved', 0.0)
        result_str = result.get('Result_Str', '')
        
        if solved == 1.0:
            successful += 1
        elif solved == 0.0:
            if 'SyntaxError' in result_str or 'invalid syntax' in result_str:
                syntax_errors += 1
            elif 'NameError' in result_str or 'is not defined' in result_str:
                import_errors += 1
            elif 'Test failed' in result_str or 'AssertionError' in result_str:
                logic_errors += 1
            else:
                other_errors += 1
        else:
            logic_errors += 1
    
    # Summary
    print("=== SUMMARY (ALL PROBLEMS) ===")
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
    print("Previous results (Algorithm Designer only):")
    print("- Success Rate: ~82.5%")
    print("- Syntax Error Rate: ~17.5%")
    print("- Logic Error Rate: ~16.3%")
    print()
    print("Current results (Programming Expert + Test Analyst):")
    print(f"- Success Rate: {successful/len(results)*100:.1f}%")
    print(f"- Syntax Error Rate: {syntax_errors/len(results)*100:.1f}%")
    print(f"- Logic Error Rate: {logic_errors/len(results)*100:.1f}%")
    print()
    
    # Calculate improvement
    success_improvement = (successful/len(results)*100) - 82.5
    syntax_improvement = 17.5 - (syntax_errors/len(results)*100)
    
    print("=== IMPROVEMENT METRICS ===")
    print(f"Success Rate Change: {success_improvement:+.1f} percentage points")
    print(f"Syntax Error Reduction: {syntax_improvement:+.1f} percentage points")
    
    if syntax_errors == 0:
        print("🎉 EXCELLENT: No syntax errors! The Programming Expert + Test Analyst combination eliminated syntax errors!")
    elif syntax_errors < len(results) * 0.1:
        print("✅ EXCELLENT: Very low syntax error rate. The combination is working very well!")
    elif syntax_errors < len(results) * 0.2:
        print("✅ GOOD: Low syntax error rate. The combination is helping!")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Still have syntax errors. May need further optimization.")

if __name__ == "__main__":
    analyze_new_pipeline_results()
