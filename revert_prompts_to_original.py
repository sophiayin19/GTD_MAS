import re

def revert_prompts_to_original():
    """Revert all agent prompts to their original, simpler versions."""
    
    prompt_file = 'GDesigner/prompt/humaneval_prompt_set.py'
    
    # Read the current prompt file
    with open(prompt_file, 'r') as f:
        content = f.read()
    
    print("Reverting prompts to original versions...")
    
    # Revert Programming Expert prompt to original
    programming_expert_old = '"Programming Expert": \n        "You are a programming expert. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. NEVER write \'Define a function\' or \'Implement a function\' - just write the actual Python code. MANDATORY SYNTAX REQUIREMENTS: Write complete, syntactically correct Python functions with proper indentation, all necessary imports, and ensure all brackets and quotes are properly closed. You will be given a function signature and its docstring by the user. You may be able to get the output results of other agents. They may have passed internal tests, but they may not be completely correct. Write your full implementation (restate the function signature). Use a Python code block to write your response. For example:\n```python\nprint(\'Hello world!\')\n```Do not include anything other than Python code blocks in your response. Do not change function names and input variable types in tasks."'
    
    programming_expert_new = '"Programming Expert": \n        "You are a programming expert. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. NEVER write \'Define a function\' or \'Implement a function\' - just write the actual Python code.  You will be given a function signature and its docstring by the user. You may be able to get the output results of other agents. They may have passed internal tests, but they may not be completely correct. Write your full implementation (restate the function signature). Use a Python code block to write your response. For example:\n```python\nprint(\'Hello world!\')\n```Do not include anything other than Python code blocks in your response. Do not change function names and input variable types in tasks."'
    
    content = content.replace(programming_expert_old, programming_expert_new)
    
    # Revert Test Analyst prompt to original
    test_analyst_old = '"Test Analyst": \n        "You are a test analyst. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. NEVER write \'Define a function\' or \'Implement a function\' - just write the actual Python code. MANDATORY SYNTAX REQUIREMENTS: Write complete, syntactically correct Python functions with proper indentation, all necessary imports, and ensure all brackets and quotes are properly closed.  You will be given a function signature and its docstring by the user. You need to provide problems in the current code or solution based on the test data and possible test feedback in the question. You need to provide additional special use cases, boundary conditions, etc. that should be paid attention to when writing code. You can point out any potential errors in the code.I hope your reply will be more concise. Preferably within fifty words. Don\'t list too many points."'
    
    test_analyst_new = '"Test Analyst": \n        "You are a test analyst. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. NEVER write \'Define a function\' or \'Implement a function\' - just write the actual Python code.  You will be given a function signature and its docstring by the user. You need to provide problems in the current code or solution based on the test data and possible test feedback in the question. You need to provide additional special use cases, boundary conditions, etc. that should be paid attention to when writing code. You can point out any potential errors in the code.I hope your reply will be more concise. Preferably within fifty words. Don\'t list too many points."'
    
    content = content.replace(test_analyst_old, test_analyst_new)
    
    # Revert Algorithm Designer prompt to original
    algorithm_designer_old = '"Algorithm Designer": \n        "You are an algorithm designer. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. NEVER write \'Define a function\' or \'Implement a function\' - just write the actual Python code.  You will be given a function signature and its docstring by the user. You need to specify the specific design of the algorithm, including the classes that may be defined and the functions used. You need to generate the detailed documentation, including explanations of the algorithm, usage instructions, and API references. When the implementation logic is complex, you can give the pseudocode logic of the main algorithm.I hope your reply will be more concise. Preferably within fifty words. Don\'t list too many points."'
    
    algorithm_designer_new = '"Algorithm Designer": \n        "You are an algorithm designer. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. NEVER write \'Define a function\' or \'Implement a function\' - just write the actual Python code.  You will be given a function signature and its docstring by the user. You need to specify the specific design of the algorithm, including the classes that may be defined and the functions used. You need to generate the detailed documentation, including explanations of the algorithm, usage instructions, and API references. When the implementation logic is complex, you can give the pseudocode logic of the main algorithm.I hope your reply will be more concise. Preferably within fifty words. Don\'t list too many points."'
    
    content = content.replace(algorithm_designer_old, algorithm_designer_new)
    
    # Write the updated content back
    with open(prompt_file, 'w') as f:
        f.write(content)
    
    print("✅ Successfully reverted all prompts to original versions!")
    print("Changes made:")
    print("- Removed MANDATORY SYNTAX REQUIREMENTS from all agents")
    print("- Removed verbose syntax requirements")
    print("- Kept the core CRITICAL requirements that were working")
    print("- Restored original, simpler prompt structure")
    print()
    print("All three agents now have the original, simpler prompts!")

if __name__ == "__main__":
    revert_prompts_to_original()
