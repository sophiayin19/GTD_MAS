import re

# Read the current file
with open('datasets/humaneval_dataset.py', 'r') as f:
    content = f.read()

# Update the humaneval_get_predict function with improved extraction
old_function = '''def humaneval_get_predict(model_response: str) -> str:
    """
    Extracts the Python code block from the model's response.
    It looks for a ```python ... ``` block and extracts the content.
    If not found, it assumes the entire response is the code.
    """
    if '```python' in model_response:
        match = re.search(r"```python\\n(.*?)\\n```", model_response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    if '```' in model_response:
        match = re.search(r"```\\n(.*?)\\n```", model_response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Fallback for code that isn't in a markdown block
    return model_response.strip()'''

new_function = '''def humaneval_get_predict(model_response: str) -> str:
    """
    Enhanced code extraction with better handling of various response formats.
    """
    # Clean the response first
    response = model_response.strip()
    
    # Try multiple extraction patterns in order of preference
    patterns = [
        # Pattern 1: ```python ... ```
        r"```python\\s*\\n(.*?)\\n```",
        # Pattern 2: ``` ... ``` (generic code block)
        r"```\\s*\\n(.*?)\\n```",
        # Pattern 3: def function with proper indentation
        r"(def\\s+\\w+\\s*\\([^)]*\\)\\s*:.*?)(?=\\n\\ndef|\\n\\nclass|\\n\\nif|\\n\\nfor|\\n\\nwhile|\\n\\n#|\\n\\n$|\\Z)",
        # Pattern 4: def function to end of response
        r"(def\\s+\\w+\\s*\\([^)]*\\)\\s*:.*)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            
            # Validate that it looks like a function
            if 'def ' in code:
                # Basic validation
                lines = code.split('\\n')
                func_line = None
                for line in lines:
                    if line.strip().startswith('def '):
                        func_line = line
                        break
                
                if func_line:
                    # Check if it has proper function signature
                    if '(' in func_line and ')' in func_line and ':' in func_line:
                        return code
    
    # Fallback: try to extract any function definition
    lines = response.split('\\n')
    code_lines = []
    in_function = False
    base_indent = None
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('def '):
            in_function = True
            base_indent = len(line) - len(line.lstrip())
            code_lines.append(line)
        elif in_function and stripped:
            current_indent = len(line) - len(line.lstrip())
            if current_indent > base_indent:
                code_lines.append(line)
            else:
                break
        elif in_function and not stripped:
            code_lines.append(line)
    
    if code_lines:
        return '\\n'.join(code_lines)
    
    # Last resort: return the whole response
    return response'''

# Update the check_correctness function to return binary scoring
old_check_correctness = '''def check_correctness(prompt: str, completion: str, test: str) -> Tuple[float, str]:
    """
    Evaluates the generated code against the provided test cases with partial credit.
    Returns a decimal score (0.0 to 1.0) based on the percentage of test cases that pass.
    """
    program = f"{prompt}\\n{completion}\\n{test}"
    try:
        exec_globals = {}
        exec(program, exec_globals)
        return 1.0, "All tests passed"
    except AssertionError as e:
        # Try to run individual test cases to see how many pass
        try:
            import ast
            tree = ast.parse(test)
            assert_count = 0
            passed_count = 0
            
            # Count total assertions
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert):
                    assert_count += 1
            
            # Try to run individual assertions
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert):
                    try:
                        # Create a minimal test environment
                        test_env = exec_globals.copy()
                        test_env['candidate'] = exec_globals.get('candidate', None)
                        
                        # Execute just this assertion
                        exec(compile(ast.Module([node], []), '<string>', 'exec'), test_env)
                        passed_count += 1
                    except:
                        pass
            
            if assert_count > 0:
                score = passed_count / assert_count
                return score, f"Partial credit: {passed_count}/{assert_count} tests passed"
            else:
                return 0.0, "No test cases found"
                
        except Exception as e:
            return 0.0, f"Failed to parse test cases: {e}"
    except Exception as e:
        return 0.0, f"Execution failed: {type(e).__name__}: {e}"'''

new_check_correctness = '''def check_correctness(prompt: str, completion: str, test: str) -> Tuple[bool, str]:
    """
    Evaluates the generated code against the provided test cases.
    Returns True if all tests pass, False otherwise.
    """
    program = f"{prompt}\\n{completion}\\n{test}"
    try:
        exec_globals = {}
        exec(program, exec_globals)
        return True, "All tests passed"
    except AssertionError as e:
        return False, f"Test failed: {e}"
    except Exception as e:
        return False, f"Execution failed: {type(e).__name__}: {e}"'''

# Replace both functions
content = content.replace(old_function, new_function)
content = content.replace(old_check_correctness, new_check_correctness)

# Write the updated content
with open('datasets/humaneval_dataset.py', 'w') as f:
    f.write(content)

print("Updated humaneval_get_predict with improved extraction and reverted check_correctness to binary scoring")
