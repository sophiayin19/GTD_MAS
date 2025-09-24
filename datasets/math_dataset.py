import re
from typing import List, Dict, Any
import sympy

def math_data_process(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processes the raw MATH dataset.
    """
    processed_dataset = []
    for record in dataset:
        answer_match = re.search(r'\\boxed{((?:.|\n)*?)}', record["solution"])
        final_answer = answer_match.group(1) if answer_match else record["answer"]
        
        processed_dataset.append({
            "task": record["problem"],
            "answer": final_answer
        })
    return processed_dataset

def math_get_predict(model_response: str) -> str:
    """
    Extracts the final answer from the model's response.
    """
    matches = re.findall(r'\\boxed{((?:.|\n)*?)}', model_response)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in model_response.strip().split('\\n')]
    if lines:
        return lines[-1]
        
    return ""

def math_check_correctness(predicted_answer: str, ground_truth: str) -> bool:
    """
    Checks if the predicted answer matches the ground truth for MATH dataset using symbolic comparison.
    """
    def normalize_answer(answer):
        """Normalize mathematical expressions for comparison."""
        # Remove common LaTeX commands
        answer = re.sub(r'\\text{[^}]*}', '', answer)
        answer = re.sub(r'\\[a-zA-Z]+\s*', '', answer)
        answer = answer.replace('\\', '').replace('{', '').replace('}', '')
        answer = answer.replace('$', '').strip()
        return answer

    try:
        pred_normalized = normalize_answer(predicted_answer)
        truth_normalized = normalize_answer(ground_truth)
        
        # Try symbolic comparison
        pred_expr = sympy.sympify(pred_normalized)
        truth_expr = sympy.sympify(truth_normalized)
        
        return sympy.simplify(pred_expr - truth_expr) == 0
    except:
        # Fall back to string comparison
        return pred_normalized.lower() == truth_normalized.lower()
