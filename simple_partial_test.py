import json
import os

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
    
    # Count solved vs failed
    solved = sum(1 for r in results if r.get('Solved', False))
    failed = len(results) - solved
    
    print(f"Solved: {solved}")
    print(f"Failed: {failed}")
    print(f"Success rate: {solved/len(results):.3f}")
    
    # Show a few failed problems to see if partial credit would help
    print("\nFirst few failed problems:")
    failed_count = 0
    for i, result in enumerate(results):
        if not result.get('Solved', False) and failed_count < 3:
            print(f"Problem {i+1}: {result.get('Result_Str', 'No result')}")
            failed_count += 1
else:
    print("No result files found")
