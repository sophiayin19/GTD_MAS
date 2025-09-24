import json
import os
import sys
sys.path.append('.')

def analyze_pass_rate_discrepancy():
    """Analyze the discrepancy between reported pass rate and actual correct answers."""
    
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
    
    print("=== PASS RATE DISCREPANCY ANALYSIS ===")
    print(f"Total problems: {len(results)}")
    print()
    
    # Calculate actual success rate
    actual_successful = 0
    for result in results:
        solved = result.get('Solved', 0.0)
        if solved == 1.0:
            actual_successful += 1
    
    actual_success_rate = actual_successful / len(results) * 100
    
    # Get the reported pass rate from the last result
    last_result = results[-1] if results else {}
    reported_pass_rate = last_result.get('Pass_Rate', 0.0) * 100
    reported_total_solved = last_result.get('Total_solved', 0.0)
    reported_total_executed = last_result.get('Total_executed', 0.0)
    
    print("=== COMPARISON ===")
    print(f"Actual Success Rate: {actual_success_rate:.1f}% ({actual_successful}/{len(results)})")
    print(f"Reported Pass Rate: {reported_pass_rate:.1f}%")
    print(f"Reported Total Solved: {reported_total_solved}")
    print(f"Reported Total Executed: {reported_total_executed}")
    print()
    
    # Calculate discrepancy
    discrepancy = abs(actual_success_rate - reported_pass_rate)
    print(f"Discrepancy: {discrepancy:.1f} percentage points")
    
    if discrepancy > 5.0:
        print("⚠️ LARGE DISCREPANCY: There's a significant difference between actual and reported rates!")
    elif discrepancy > 1.0:
        print("⚠️ MODERATE DISCREPANCY: There's a noticeable difference between actual and reported rates.")
    else:
        print("✅ MINIMAL DISCREPANCY: The rates are very close.")
    
    print()
    
    # Analyze individual results to understand the discrepancy
    print("=== DETAILED ANALYSIS ===")
    print("First 10 problems:")
    print("-" * 50)
    
    for i, result in enumerate(results[:10], 1):
        solved = result.get('Solved', 0.0)
        result_str = result.get('Result_Str', '')
        
        print(f"Problem {i}:")
        print(f"  Solved: {solved}")
        print(f"  Result: {result_str[:100]}..." if len(result_str) > 100 else f"  Result: {result_str}")
        print()
    
    # Check if there are any partial successes
    partial_successes = 0
    for result in results:
        solved = result.get('Solved', 0.0)
        if 0 < solved < 1.0:
            partial_successes += 1
    
    print(f"Partial successes (0 < score < 1): {partial_successes}")
    
    # Check if the issue is with partial credit scoring
    if partial_successes > 0:
        print("ℹ️ Note: Partial credit scoring is being used, which might explain the discrepancy.")
        print("The 'Solved' field shows partial credit, while 'Pass_Rate' might be calculated differently.")
    
    print()
    print("=== RECOMMENDATIONS ===")
    if discrepancy > 5.0:
        print("1. Check how 'Pass_Rate' is calculated in the code")
        print("2. Verify if partial credit is being handled consistently")
        print("3. Ensure 'Total_solved' and 'Total_executed' are calculated correctly")
    else:
        print("The discrepancy is minimal and likely due to rounding or partial credit differences.")

if __name__ == "__main__":
    analyze_pass_rate_discrepancy()
