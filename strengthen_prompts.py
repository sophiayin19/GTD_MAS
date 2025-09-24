import re

def strengthen_prompts():
    """Strengthen the prompts to prevent descriptions and ensure code generation."""
    
    prompt_file = 'GDesigner/prompt/humaneval_prompt_set.py'
    
    with open(prompt_file, 'r') as f:
        content = f.read()
    
    # Add even stronger requirements to prevent descriptions
    stronger_requirements = "CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. NEVER write 'Define a function' or 'Implement a function' - just write the actual Python code. "
    
    # Update each agent prompt
    agents = ['Programming_Expert', 'Algorithm_Designer', 'Test_Analyst', 'Bug_Fixer']
    
    for agent in agents:
        # Find and replace the current CRITICAL requirements with stronger ones
        old_pattern = f'"You are a {agent.lower().replace("_", " ")}. CRITICAL: Generate ONLY executable Python code with proper syntax. Use def keyword, correct indentation, and valid Python. Do NOT generate pseudo-code or descriptions. '
        new_pattern = f'"You are a {agent.lower().replace("_", " ")}. {stronger_requirements}'
        
        content = content.replace(old_pattern, new_pattern)
    
    # Write the updated content back
    with open(prompt_file, 'w') as f:
        f.write(content)
    
    print("✓ Strengthened prompts to prevent descriptions")
    print("✓ Added explicit prohibition against 'Define a function' and 'Implement a function'")
    print("✓ Emphasized writing actual Python code only")

if __name__ == "__main__":
    strengthen_prompts()
