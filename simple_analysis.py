import json
import os

# Check the latest result file
latest_file = 'result/gtd_humaneval/gpt-4o-mini_2025-09-07-19-18-07.json'

with open(latest_file, 'r') as f:
    results = json.load(f)

print(f"Total problems: {len(results)}")
print(f"Keys in first result: {list(results[0].keys())}")

# Show first result
print("\nFirst result:")
print(json.dumps(results[0], indent=2)[:500] + "...")

# Count successful vs failed
successful = sum(1 for r in results if r.get('score', 0) > 0)
failed = len(results) - successful
print(f"\nSummary:")
print(f"Successful: {successful}/{len(results)}")
print(f"Failed: {failed}/{len(results)}")
print(f"Success rate: {successful/len(results)*100:.1f}%")
