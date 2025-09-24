import json
import sys
sys.path.append('.')

from datasets.humaneval_dataset import humaneval_get_predict

def test_improved_prompts():
    """Test if the prompt improvements are working."""
    
    latest_file = 'result/gtd_humaneval/gpt-4o-mini_2025-09-07-19-18-07.json'
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print("=== TESTING IMPROVED PROMPTS ===")
    
    # Check if we have any recent results with improved prompts
    # Look for results that might have been generated after the prompt improvements
    
    # For now, let's check the current prompt file to see if it was updated
    try:
        with open('GDesigner/prompt/humaneval_prompt_set.py', 'r') as f:
            prompt_content = f.read()
        
        if 'CRITICAL REQUIREMENTS' in prompt_content:
            print("✓ Prompt improvements were applied successfully")
            print("✓ Found 'CRITICAL REQUIREMENTS' in prompt file")
        else:
            print("✗ Prompt improvements were not applied")
            print("✗ 'CRITICAL REQUIREMENTS' not found in prompt file")
            
    except Exception as e:
        print(f"Error reading prompt file: {e}")
    
    # Check the current results to see if there are any improvements
    print(f"\nCurrent results summary:")
    successful = sum(1 for r in results if r.get('Solved', False))
    total = len(results)
    print(f"Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    # Check syntax error rate
    syntax_errors = sum(1 for r in results if not r.get('Solved', False) and 'SyntaxError' in r.get('Result_Str', ''))
    print(f"Syntax error rate: {syntax_errors}/{total} ({syntax_errors/total*100:.1f}%)")

if __name__ == "__main__":
    test_improved_prompts()
