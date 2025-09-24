import re

def fix_prompts():
    prompt_file = 'GDesigner/prompt/humaneval_prompt_set.py'
    
    with open(prompt_file, 'r') as f:
        content = f.read()
    
    # Add critical requirements to Programming Expert
    programming_expert_old = '"Programming Expert": \n        "You are a programming expert. '
    programming_expert_new = '"Programming Expert": \n        "You are a programming expert. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. '
    
    if programming_expert_old in content:
        content = content.replace(programming_expert_old, programming_expert_new)
        print("✓ Updated Programming Expert prompt")
    else:
        print("✗ Could not find Programming Expert prompt")
    
    # Add critical requirements to Algorithm Designer
    algo_designer_old = '"Algorithm Designer": \n        "You are an algorithm designer. '
    algo_designer_new = '"Algorithm Designer": \n        "You are an algorithm designer. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. '
    
    if algo_designer_old in content:
        content = content.replace(algo_designer_old, algo_designer_new)
        print("✓ Updated Algorithm Designer prompt")
    else:
        print("✗ Could not find Algorithm Designer prompt")
    
    # Add critical requirements to Test Analyst
    test_analyst_old = '"Test Analyst": \n        "You are a test analyst. '
    test_analyst_new = '"Test Analyst": \n        "You are a test analyst. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. '
    
    if test_analyst_old in content:
        content = content.replace(test_analyst_old, test_analyst_new)
        print("✓ Updated Test Analyst prompt")
    else:
        print("✗ Could not find Test Analyst prompt")
    
    # Add critical requirements to Bug Fixer
    bug_fixer_old = '"Bug Fixer": \n        "You are a bug fixer. '
    bug_fixer_new = '"Bug Fixer": \n        "You are a bug fixer. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. '
    
    if bug_fixer_old in content:
        content = content.replace(bug_fixer_old, bug_fixer_new)
        print("✓ Updated Bug Fixer prompt")
    else:
        print("✗ Could not find Bug Fixer prompt")
    
    # Write the updated content back
    with open(prompt_file, 'w') as f:
        f.write(content)
    
    print("✓ Updated all agent prompts with critical requirements")

if __name__ == "__main__":
    fix_prompts()
