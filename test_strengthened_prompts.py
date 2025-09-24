import sys
sys.path.append('.')

from GDesigner.prompt.humaneval_prompt_set import HumanEvalPromptSet

def test_strengthened_prompts():
    """Test the strengthened prompts to see if they prevent descriptions."""
    
    prompt_set = HumanEvalPromptSet()
    
    print("=== TESTING STRENGTHENED PROMPTS ===")
    
    # Test each agent prompt
    agents = ['Programming_Expert', 'Algorithm_Designer', 'Test_Analyst', 'Bug_Fixer']
    
    for agent in agents:
        prompt = prompt_set.get_constraint(agent)
        print(f"\n{agent}:")
        print(f"Prompt: {prompt[:200]}...")
        
        # Check if the strengthened requirements are present
        if "NEVER write 'Define a function'" in prompt:
            print("✓ Strengthened requirements found")
        else:
            print("✗ Strengthened requirements not found")
        
        # Check if it still has the original CRITICAL requirements
        if "CRITICAL: Generate ONLY executable Python code" in prompt:
            print("✓ Original CRITICAL requirements found")
        else:
            print("✗ Original CRITICAL requirements not found")

if __name__ == "__main__":
    test_strengthened_prompts()
