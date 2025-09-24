import json
import os
import subprocess
import time

def test_three_combinations():
    """Test three specific agent combinations on 40 questions."""
    
    # Define the three combinations to test
    agent_combinations = [
        (["Programming_Expert", "Algorithm_Designer"], "2 2", 4, "2 Programming Experts + 2 Algorithm Designers"),
        (["Programming_Expert", "Bug_Fixer"], "2 2", 4, "2 Programming Experts + 2 Bug Fixers"),
        (["Algorithm_Designer", "Test_Analyst"], "2 2", 4, "2 Algorithm Designers + 2 Test Analysts"),
    ]
    
    print("=== TESTING THREE SPECIFIC COMBINATIONS ON 40 QUESTIONS ===")
    print("Testing each combination on 40 problems...")
    print()
    
    results = []
    
    for i, (agent_names, agent_nums, agent_count, description) in enumerate(agent_combinations, 1):
        print(f"Test {i}/3: {description}")
        print(f"Agents: {agent_names}")
        print(f"Count: {agent_count}")
        
        # Create test dataset with 40 problems
        test_file = f"datasets/humaneval/humaneval-test-{i}.jsonl"
        subprocess.run(f"head -40 datasets/humaneval/humaneval-py.jsonl > {test_file}", shell=True)
        
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
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1200)  # Increased timeout for 40 problems
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
                    
                    # Count error types
                    syntax_errors = sum(1 for r in test_results if r.get('Solved', 0) == 0.0 and ('SyntaxError' in r.get('Result_Str', '') or 'invalid syntax' in r.get('Result_Str', '')))
                    logic_errors = sum(1 for r in test_results if r.get('Solved', 0) == 0.0 and ('Test failed' in r.get('Result_Str', '') or 'AssertionError' in r.get('Result_Str', '')))
                    other_errors = sum(1 for r in test_results if r.get('Solved', 0) == 0.0 and not ('SyntaxError' in r.get('Result_Str', '') or 'invalid syntax' in r.get('Result_Str', '') or 'Test failed' in r.get('Result_Str', '') or 'AssertionError' in r.get('Result_Str', '')))
                    
                    results.append({
                        'description': description,
                        'agents': agent_names,
                        'count': agent_count,
                        'success_rate': success_rate,
                        'successful': successful,
                        'total': len(test_results),
                        'syntax_errors': syntax_errors,
                        'logic_errors': logic_errors,
                        'other_errors': other_errors,
                        'time': end_time - start_time
                    })
                    
                    print(f"✅ Success Rate: {success_rate:.1f}% ({successful}/{len(test_results)})")
                    print(f"❌ Syntax Errors: {syntax_errors}")
                    print(f"❌ Logic Errors: {logic_errors}")
                    print(f"❌ Other Errors: {other_errors}")
                    print(f"⏱️ Time: {end_time - start_time:.1f}s")
                else:
                    print("❌ No result file found")
            else:
                print(f"❌ Test failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ Test timed out")
        except Exception as e:
            print(f"❌ Test error: {e}")
        
        print("-" * 60)
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
        print(f"   Syntax Errors: {result['syntax_errors']}")
        print(f"   Logic Errors: {result['logic_errors']}")
        print(f"   Other Errors: {result['other_errors']}")
        print(f"   Time: {result['time']:.1f}s")
        print()
    
    # Find the best combination
    if results:
        best = results[0]
        print("=== BEST COMBINATION ===")
        print(f" {best['description']}")
        print(f"Success Rate: {best['success_rate']:.1f}%")
        print(f"Syntax Errors: {best['syntax_errors']}")
        print(f"Logic Errors: {best['logic_errors']}")
        print(f"Other Errors: {best['other_errors']}")
        print(f"Agents: {best['agents']}")
        print(f"Count: {best['count']}")
        print()
        print("This combination should be tested on the full 80-problem dataset!")
        
        # Show comparison with current best
        print("=== COMPARISON WITH CURRENT BEST ===")
        print("Current best (Programming Expert + Test Analyst): 72.5% success rate")
        print(f"Best from this test: {best['success_rate']:.1f}% success rate")
        improvement = best['success_rate'] - 72.5
        if improvement > 0:
            print(f"Improvement: +{improvement:.1f} percentage points 🚀")
        else:
            print(f"Difference: {improvement:.1f} percentage points")

if __name__ == "__main__":
    test_three_combinations()
