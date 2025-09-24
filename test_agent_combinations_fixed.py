import json
import os
import subprocess
import time

def test_agent_combinations():
    """Test different agent combinations to see which performs best."""
    
    # Define different agent combinations to test
    # Each combination is a list of individual agent names
    agent_combinations = [
        # Single agents
        (["Programming_Expert"], "1", 1, "Single Programming Expert"),
        (["Algorithm_Designer"], "1", 1, "Single Algorithm Designer"),
        (["Test_Analyst"], "1", 1, "Single Test Analyst"),
        (["Bug_Fixer"], "1", 1, "Single Bug Fixer"),
        
        # Two-agent combinations
        (["Programming_Expert", "Algorithm_Designer"], "1 1", 2, "Programming Expert + Algorithm Designer"),
        (["Programming_Expert", "Test_Analyst"], "1 1", 2, "Programming Expert + Test Analyst"),
        (["Programming_Expert", "Bug_Fixer"], "1 1", 2, "Programming Expert + Bug Fixer"),
        (["Algorithm_Designer", "Test_Analyst"], "1 1", 2, "Algorithm Designer + Test Analyst"),
        
        # Three-agent combinations
        (["Programming_Expert", "Algorithm_Designer", "Test_Analyst"], "1 1 1", 3, "Programming Expert + Algorithm Designer + Test Analyst"),
        (["Programming_Expert", "Test_Analyst", "Bug_Fixer"], "1 1 1", 3, "Programming Expert + Test Analyst + Bug Fixer"),
        (["Algorithm_Designer", "Test_Analyst", "Bug_Fixer"], "1 1 1", 3, "Algorithm Designer + Test Analyst + Bug Fixer"),
        
        # Four-agent combination
        (["Programming_Expert", "Algorithm_Designer", "Test_Analyst", "Bug_Fixer"], "1 1 1 1", 4, "All Four Agents"),
    ]
    
    print("=== TESTING DIFFERENT AGENT COMBINATIONS ===")
    print("Testing on 10 problems each to find the best combination...")
    print()
    
    results = []
    
    for i, (agent_names, agent_nums, agent_count, description) in enumerate(agent_combinations, 1):
        print(f"Test {i}/{len(agent_combinations)}: {description}")
        print(f"Agents: {agent_names}")
        print(f"Count: {agent_count}")
        
        # Create test dataset with 10 problems
        test_file = f"datasets/humaneval/humaneval-test-{i}.jsonl"
        subprocess.run(f"head -10 datasets/humaneval/humaneval-py.jsonl > {test_file}", shell=True)
        
        # Convert agent names list to space-separated string for the command
        agent_names_str = " ".join(agent_names)
        
        # Run the test
        cmd = f"""python experiments/run_humaneval.py \\
            --mode GTD \\
            --llm_name gpt-4o-mini \\
            --domain humaneval \\
            --agent_names {agent_names_str} \\
            --agent_nums {agent_nums} \\
            --dataset_json {test_file} \\
            --gtd-proxy-model-path trained_models/proxy_model_humaneval.pth \\
            --gtd-diffusion-model-path trained_models/diffusion_model_humaneval.pth"""
        
        print("Running test...")
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            end_time = time.time()
            
            if result.returncode == 0:
                # Find the result file
                result_files = []
                if os.path.exists('result/gtd_humaneval'):
                    for file in os.listdir('result/gtd_humaneval'):
                        if file.endswith('.json'):
                            result_files.append(os.path.join('result/gtd_humaneval', file))
                
                if result_files:
                    latest_file = max(result_files, key=os.path.getmtime)
                    
                    with open(latest_file, 'r') as f:
                        test_results = json.load(f)
                    
                    successful = sum(1 for r in test_results if r.get('Solved', 0) == 1.0)
                    success_rate = successful / len(test_results) * 100
                    
                    results.append({
                        'description': description,
                        'agents': agent_names,
                        'count': agent_count,
                        'success_rate': success_rate,
                        'successful': successful,
                        'total': len(test_results),
                        'time': end_time - start_time
                    })
                    
                    print(f"✅ Success Rate: {success_rate:.1f}% ({successful}/{len(test_results)})")
                    print(f"⏱️ Time: {end_time - start_time:.1f}s")
                else:
                    print("❌ No result file found")
            else:
                print(f"❌ Test failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ Test timed out")
        except Exception as e:
            print(f"❌ Test error: {e}")
        
        print("-" * 50)
        print()
        
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)
    
    # Sort results by success rate
    results.sort(key=lambda x: x['success_rate'], reverse=True)
    
    print("=== FINAL RESULTS (Ranked by Success Rate) ===")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['description']}")
        print(f"   Success Rate: {result['success_rate']:.1f}% ({result['successful']}/{result['total']})")
        print(f"   Agents: {result['agents']}")
        print(f"   Time: {result['time']:.1f}s")
        print()
    
    # Find the best combination
    if results:
        best = results[0]
        print("=== BEST COMBINATION ===")
        print(f"�� {best['description']}")
        print(f"Success Rate: {best['success_rate']:.1f}%")
        print(f"Agents: {best['agents']}")
        print(f"Count: {best['count']}")
        print()
        print("This combination should be tested on the full 80-problem dataset!")

if __name__ == "__main__":
    test_agent_combinations()
