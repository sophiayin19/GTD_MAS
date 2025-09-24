import json
import sys
sys.path.append('.')

from GDesigner.prompt.humaneval_prompt_set import HumanEvalPromptSet

def test_strengthened_prompts():
    """Test the strengthened prompts on a small sample."""
    
    prompt_set = HumanEvalPromptSet()
    
    print("=== TESTING STRENGTHENED PROMPTS ===")
    
    # Test each agent prompt
    agents = ['Programming Expert', 'Algorithm Designer', 'Test Analyst', 'Bug Fixer']
    
    for agent in agents:
        prompt = prompt_set.get_constraint(agent)
        print(f"\n{agent}:")
        print(f"Prompt: {prompt[:200]}...")
        
        # Check if the strengthened requirements are present
        if "NEVER write 'Define a function'" in prompt:
            print("✓ Strengthened requirements found")
        else:
            print("✗ Strengthened requirements not found")

if __name__ == "__main__":
    test_strengthened_prompts()
