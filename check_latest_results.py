import json
import os

def check_latest_results():
    """Check the latest results for partial credit scores."""
    
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
        
        # Look for partial credit scores (between 0 and 1, but not exactly 0 or 1)
        partial_scores = []
        for i, result in enumerate(results):
            # Check if there's a score field that might show partial credit
            if 'score' in result:
                score = result['score']
                if isinstance(score, float) and 0 < score < 1:
                    partial_scores.append((i+1, score))
        
        if partial_scores:
            print(f"\nFound {len(partial_scores)} problems with partial credit:")
            for prob_num, score in partial_scores[:5]:  # Show first 5
                print(f"  Problem {prob_num}: {score:.3f}")
        else:
            print("\nNo partial credit scores found (all problems are 0.0 or 1.0)")
            print("This suggests most problems are either completely correct or completely wrong")
        
        # Show overall statistics
        solved = sum(1 for r in results if r.get('Solved', False))
        print(f"\nOverall: {solved}/{len(results)} solved ({solved/len(results)*100:.1f}%)")
    else:
        print("No result files found")

if __name__ == "__main__":
    check_latest_results()
