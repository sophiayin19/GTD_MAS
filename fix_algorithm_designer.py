import re

def fix_algorithm_designer():
    """Fix the Algorithm Designer prompt to include strengthened requirements."""
    
    prompt_file = 'GDesigner/prompt/humaneval_prompt_set.py'
    
    with open(prompt_file, 'r') as f:
        content = f.read()
    
    # Find and replace the Algorithm Designer prompt
    old_pattern = '"Algorithm Designer": \n        "You are an algorithm designer. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions.  '
    new_pattern = '"Algorithm Designer": \n        "You are an algorithm designer. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. NEVER write \'Define a function\' or \'Implement a function\' - just write the actual Python code. '
    
    content = content.replace(old_pattern, new_pattern)
    
    # Write the updated content back
    with open(prompt_file, 'w') as f:
        f.write(content)
    
    print("✓ Fixed Algorithm Designer prompt with strengthened requirements")

if __name__ == "__main__":
    fix_algorithm_designer()
