import re

# Read the current file
with open('datasets/humaneval_dataset.py', 'r') as f:
    content = f.read()

# Revert to the original simple extraction
old_function = '''def humaneval_get_predict(model_response: str) -> str:
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

new_function = '''def humaneval_get_predict(model_response: str) -> str:
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

# Replace the function
content = content.replace(old_function, new_function)

# Write the updated content
with open('datasets/humaneval_dataset.py', 'w') as f:
    f.write(content)

print("Reverted to original simple extraction method while keeping binary scoring")
