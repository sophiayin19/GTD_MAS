import json
import os
import sys
sys.path.append('.')

def analyze_all_results():
    """Analyze all results to count solved problems and syntax errors."""
    
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
    
    print("=== COMPREHENSIVE RESULTS ANALYSIS ===")
    print(f"Total problems: {len(results)}")
    print()
    
    # Categorize all results
    successful = 0
    syntax_errors = 0
    logic_errors = 0
    import_errors = 0
    other_errors = 0
    partial_successes = 0
    
    # Track specific error types
    error_details = {
        'SyntaxError': 0,
        'NameError': 0,
        'AssertionError': 0,
        'Test failed': 0,
        'Execution failed': 0,
        'Other': 0
    }
    
    print("=== PROBLEM-BY-PROBLEM ANALYSIS ===")
    print("Problem | Status | Error Type")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        solved = result.get('Solved', 0.0)
        result_str = result.get('Result_Str', '')
        
        if solved == 1.0:
            status = "✅ SOLVED"
            successful += 1
            error_type = "None"
        elif solved == 0.0:
            if 'SyntaxError' in result_str or 'invalid syntax' in result_str:
                status = "❌ SYNTAX ERROR"
                syntax_errors += 1
                error_details['SyntaxError'] += 1
                error_type = "SyntaxError"
            elif 'NameError' in result_str or 'is not defined' in result_str:
                status = "❌ IMPORT ERROR"
                import_errors += 1
                error_details['NameError'] += 1
                error_type = "NameError"
            elif 'Test failed' in result_str or 'AssertionError' in result_str:
                status = "❌ LOGIC ERROR"
                logic_errors += 1
                error_details['AssertionError'] += 1
                error_type = "LogicError"
            elif 'Execution failed' in result_str:
                status = "❌ EXECUTION ERROR"
                other_errors += 1
                error_details['Execution failed'] += 1
                error_type = "ExecutionError"
            else:
                status = "❌ OTHER ERROR"
                other_errors += 1
                error_details['Other'] += 1
                error_type = "Other"
        else:
            status = "⚠️ PARTIAL SUCCESS"
            partial_successes += 1
            error_type = "Partial"
        
        print(f"{i:3d}     | {status} | {error_type}")
    
    print()
    print("=== SUMMARY STATISTICS ===")
    print(f"✅ Solved Problems: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"❌ Syntax Errors: {syntax_errors}/{len(results)} ({syntax_errors/len(results)*100:.1f}%)")
    print(f"❌ Logic Errors: {logic_errors}/{len(results)} ({logic_errors/len(results)*100:.1f}%)")
    print(f"❌ Import Errors: {import_errors}/{len(results)} ({import_errors/len(results)*100:.1f}%)")
    print(f"❌ Other Errors: {other_errors}/{len(results)} ({other_errors/len(results)*100:.1f}%)")
    print(f"⚠️ Partial Successes: {partial_successes}/{len(results)} ({partial_successes/len(results)*100:.1f}%)")
    print()
    
    print("=== ERROR BREAKDOWN ===")
    for error_type, count in error_details.items():
        if count > 0:
            print(f"{error_type}: {count} problems ({count/len(results)*100:.1f}%)")
    print()
    
    print("=== SUCCESS RATE ANALYSIS ===")
    print(f"Overall Success Rate: {successful/len(results)*100:.1f}%")
    print(f"Syntax Error Rate: {syntax_errors/len(results)*100:.1f}%")
    print(f"Logic Error Rate: {logic_errors/len(results)*100:.1f}%")
    print()
    
    # Show which problems had syntax errors
    if syntax_errors > 0:
        print("=== PROBLEMS WITH SYNTAX ERRORS ===")
        syntax_problems = []
        for i, result in enumerate(results, 1):
            solved = result.get('Solved', 0.0)
            result_str = result.get('Result_Str', '')
            if solved == 0.0 and ('SyntaxError' in result_str or 'invalid syntax' in result_str):
                syntax_problems.append(i)
        
        print(f"Problems with syntax errors: {syntax_problems}")
        print(f"Total syntax error problems: {len(syntax_problems)}")
        print()
    
    # Performance assessment
    print("=== PERFORMANCE ASSESSMENT ===")
    if syntax_errors == 0:
        print("🎉 EXCELLENT: No syntax errors! The MANDATORY syntax requirements worked perfectly!")
    elif syntax_errors < len(results) * 0.1:
        print("✅ EXCELLENT: Very low syntax error rate (<10%). The syntax requirements are working well!")
    elif syntax_errors < len(results) * 0.2:
        print("✅ GOOD: Low syntax error rate (<20%). The syntax requirements are helping!")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Still have significant syntax errors. May need further prompt optimization.")
    
    if successful/len(results) >= 0.8:
        print("🚀 OUTSTANDING: Success rate is 80% or higher!")
    elif successful/len(results) >= 0.7:
        print("🎯 GREAT: Success rate is 70% or higher!")
    elif successful/len(results) >= 0.6:
        print("�� GOOD: Success rate is 60% or higher!")
    else:
        print("📈 ROOM FOR IMPROVEMENT: Success rate is below 60%.")

if __name__ == "__main__":
    analyze_all_results()
