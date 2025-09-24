import json
import os

# Check the structure of the latest result file
latest_file = 'result/gtd_humaneval/gpt-4o-mini_2025-09-08-02-54-19.json'

with open(latest_file, 'r') as f:
    results = json.load(f)

print(f"Total results: {len(results)}")
print(f"Type of results: {type(results)}")

if len(results) > 0:
    print(f"First result keys: {list(results[0].keys())}")
    print(f"First result: {results[0]}")
    
    # Check if there are any successful results
    successful_count = 0
    for i, result in enumerate(results):
        if 'score' in result and result['score'] == 1.0:
            successful_count += 1
        elif 'score' in result and result['score'] > 0:
            print(f"Partial success at index {i}: {result}")
    
    print(f"Successful results: {successful_count}")
    
    # Check error messages
    error_count = 0
    for i, result in enumerate(results):
        if 'error_message' in result and result['error_message']:
            error_count += 1
            if error_count <= 5:  # Show first 5 errors
                print(f"Error at index {i}: {result['error_message']}")
    
    print(f"Results with error messages: {error_count}")
