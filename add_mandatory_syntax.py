import re

def add_mandatory_syntax():
    """Add MANDATORY in front of syntax requirements."""
    
    prompt_file = 'GDesigner/prompt/humaneval_prompt_set.py'
    
    # Read the current prompt file
    with open(prompt_file, 'r') as f:
        content = f.read()
    
    print("Adding MANDATORY in front of syntax requirements...")
    
    # Add MANDATORY to Programming Expert prompt
    programming_expert_old = 'SYNTAX REQUIREMENTS: Write complete, syntactically correct Python functions with proper indentation, all necessary imports, and ensure all brackets and quotes are properly closed.'
    
    programming_expert_new = 'MANDATORY SYNTAX REQUIREMENTS: Write complete, syntactically correct Python functions with proper indentation, all necessary imports, and ensure all brackets and quotes are properly closed.'
    
    content = content.replace(programming_expert_old, programming_expert_new)
    
    # Add MANDATORY to Test Analyst prompt
    test_analyst_old = 'SYNTAX REQUIREMENTS: Write complete, syntactically correct Python functions with proper indentation, all necessary imports, and ensure all brackets and quotes are properly closed.'
    
    test_analyst_new = 'MANDATORY SYNTAX REQUIREMENTS: Write complete, syntactically correct Python functions with proper indentation, all necessary imports, and ensure all brackets and quotes are properly closed.'
    
    content = content.replace(test_analyst_old, test_analyst_new)
    
    # Write the updated content back
    with open(prompt_file, 'w') as f:
        f.write(content)
    
    print("✅ Successfully added MANDATORY in front of syntax requirements!")
    print("Changes made:")
    print("- Programming Expert: SYNTAX REQUIREMENTS → MANDATORY SYNTAX REQUIREMENTS")
    print("- Test Analyst: SYNTAX REQUIREMENTS → MANDATORY SYNTAX REQUIREMENTS")
    print()
    print("The prompts now emphasize that syntax requirements are MANDATORY!")

if __name__ == "__main__":
    add_mandatory_syntax()
