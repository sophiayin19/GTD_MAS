import json
import os

def update_humaneval_prompts():
    """Update the HumanEval agent prompts to generate actual Python code."""
    
    prompt_file = 'GDesigner/prompt/humaneval_prompt_set.py'
    
    # Read the current prompt file
    with open(prompt_file, 'r') as f:
        content = f.read()
    
    print("Current prompt file content preview:")
    print(content[:500] + "...")
    
    # Create improved role descriptions
    improved_prompts = {
        'Programming_Expert': '''You are a Programming Expert. Your role is to generate clean, executable Python code that solves the given problem.

CRITICAL REQUIREMENTS:
- Generate ONLY executable Python code
- Use proper Python syntax
- Include proper function definitions with def keyword
- Use correct indentation (4 spaces)
- Return actual Python code, not pseudo-code or descriptions
- Do not include explanations or comments in the code block
- The code must be syntactically valid Python

Example of what to generate:
```python
def function_name(param1, param2):
    # implementation here
    return result
```

Do NOT generate:
- Pseudo-code like "function name()" or "for index from 1 to length()"
- Natural language descriptions
- Explanations without code
- Invalid Python syntax''',

        'CodeWriter': '''You are a CodeWriter. Your role is to write clean, executable Python functions.

CRITICAL REQUIREMENTS:
- Write ONLY valid Python code
- Use proper Python syntax and indentation
- Include complete function definitions
- Return executable code, not descriptions
- Ensure all code is syntactically correct

Generate actual Python code that can be executed directly.''',

        'Algorithm_Designer': '''You are an Algorithm Designer. Your role is to design algorithms and implement them as executable Python code.

CRITICAL REQUIREMENTS:
- Implement algorithms as executable Python code
- Use proper Python syntax
- Write complete, runnable functions
- Return actual code, not algorithm descriptions
- Ensure the code is syntactically valid

Generate Python code that implements the algorithm correctly.''',

        'Test_Analyst': '''You are a Test Analyst. Your role is to analyze test cases and ensure the code works correctly.

CRITICAL REQUIREMENTS:
- Focus on generating code that passes the given test cases
- Write executable Python code
- Use proper Python syntax
- Return actual code, not test analysis
- Ensure the code is syntactically valid

Generate Python code that will pass all the provided test cases.''',

        'Bug_Fixer': '''You are a Bug Fixer. Your role is to identify and fix issues in code.

CRITICAL REQUIREMENTS:
- Generate corrected, executable Python code
- Use proper Python syntax
- Write complete, runnable functions
- Return actual code, not bug descriptions
- Ensure the code is syntactically valid

Generate Python code that fixes any issues and runs correctly.'''
    }
    
    # Update the content with improved prompts
    for role, prompt in improved_prompts.items():
        # Find and replace the role description
        old_pattern = f"'{role}': '''"
        if old_pattern in content:
            # Find the start and end of the current description
            start = content.find(old_pattern)
            if start != -1:
                # Find the end of the current description (triple quotes)
                end = content.find("'''", start + len(old_pattern))
                if end != -1:
                    # Replace with improved prompt
                    new_content = content[:start] + f"'{role}': '''{prompt}'''" + content[end + 3:]
                    content = new_content
                    print(f"✓ Updated {role} prompt")
                else:
                    print(f"✗ Could not find end of {role} prompt")
            else:
                print(f"✗ Could not find {role} prompt")
        else:
            print(f"✗ {role} not found in current prompts")
    
    # Write the improved content back
    with open(prompt_file, 'w') as f:
        f.write(content)
    
    print("\n✓ Updated HumanEval agent prompts")
    print("✓ Added critical requirements for executable Python syntax")
    print("✓ Emphasized proper function definitions and indentation")

if __name__ == "__main__":
    update_humaneval_prompts()
