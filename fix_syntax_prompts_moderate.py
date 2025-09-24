import re

def fix_syntax_prompts_moderate():
    """Add moderate syntax requirements - more than simple, less than verbose."""
    
    prompt_file = 'GDesigner/prompt/humaneval_prompt_set.py'
    
    # Read the current prompt file
    with open(prompt_file, 'r') as f:
        content = f.read()
    
    print("Adding moderate syntax requirements...")
    
    # Moderate Programming Expert prompt
    programming_expert_old = '"Programming Expert": \n        "You are a programming expert. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. NEVER write \'Define a function\' or \'Implement a function\' - just write the actual Python code. MANDATORY: Write complete, syntactically correct Python functions. You will be given a function signature and its docstring by the user. You may be able to get the output results of other agents. They may have passed internal tests, but they may not be completely correct. Write your full implementation (restate the function signature). Use a Python code block to write your response. For example:\n```python\nprint(\'Hello world!\')\n```Do not include anything other than Python code blocks in your response. Do not change function names and input variable types in tasks."'
    
    programming_expert_new = '"Programming Expert": \n        "You are a programming expert. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. NEVER write \'Define a function\' or \'Implement a function\' - just write the actual Python code. SYNTAX REQUIREMENTS: Write complete, syntactically correct Python functions with proper indentation, all necessary imports, and ensure all brackets and quotes are properly closed. You will be given a function signature and its docstring by the user. You may be able to get the output results of other agents. They may have passed internal tests, but they may not be completely correct. Write your full implementation (restate the function signature). Use a Python code block to write your response. For example:\n```python\nprint(\'Hello world!\')\n```Do not include anything other than Python code blocks in your response. Do not change function names and input variable types in tasks."'
    
    content = content.replace(programming_expert_old, programming_expert_new)
    
    # Moderate Test Analyst prompt
    test_analyst_old = '"Test Analyst": \n        "You are a test analyst. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. NEVER write \'Define a function\' or \'Implement a function\' - just write the actual Python code. MANDATORY: Write complete, syntactically correct Python functions.  You will be given a function signature and its docstring by the user. You need to provide problems in the current code or solution based on the test data and possible test feedback in the question. You need to provide additional special use cases, boundary conditions, etc. that should be paid attention to when writing code. You can point out any potential errors in the code.I hope your reply will be more concise. Preferably within fifty words. Don\'t list too many points."'
    
    test_analyst_new = '"Test Analyst": \n        "You are a test analyst. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. NEVER write \'Define a function\' or \'Implement a function\' - just write the actual Python code. SYNTAX REQUIREMENTS: Write complete, syntactically correct Python functions with proper indentation, all necessary imports, and ensure all brackets and quotes are properly closed.  You will be given a function signature and its docstring by the user. You need to provide problems in the current code or solution based on the test data and possible test feedback in the question. You need to provide additional special use cases, boundary conditions, etc. that should be paid attention to when writing code. You can point out any potential errors in the code.I hope your reply will be more concise. Preferably within fifty words. Don\'t list too many points."'
    
    content = content.replace(test_analyst_old, test_analyst_new)
    
    # Write the updated content back
    with open(prompt_file, 'w') as f:
        f.write(content)
    
    print("✅ Successfully added moderate syntax requirements!")
    print("Changes made:")
    print("- Added SYNTAX REQUIREMENTS with key points:")
    print("  * Write complete, syntactically correct Python functions")
    print("  * Use proper indentation")
    print("  * Include all necessary imports")
    print("  * Ensure all brackets and quotes are properly closed")
    print("- Kept the core CRITICAL requirements")
    print("- More detailed than simple, less overwhelming than verbose")
    print()
    print("The prompts now have moderate verbosity for syntax requirements!")

if __name__ == "__main__":
    fix_syntax_prompts_moderate()
