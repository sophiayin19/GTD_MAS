from typing import List, Dict, Any
import re

def multiarith_data_process(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processes the raw MultiArith dataset.
    """
    processed_dataset = []
    for record in dataset:
        processed_dataset.append({
            "task": record["question"],
            "answer": str(record["final_ans"])
        })
    return processed_dataset

def multiarith_get_predict(model_response: str) -> str:
    """
    Extracts the final numerical answer from the model's response.
    It looks for the last number in the response.
    """
    # Find all numbers (including decimals and negatives)
    numbers = re.findall(r"[-+]?\d*\.?\d+", model_response)
    if numbers:
        return numbers[-1]
    return ""

def multiarith_check_correctness(predicted_answer: str, ground_truth: str) -> bool:
    """
    Checks if the predicted answer matches the ground truth for MultiArith dataset.
    """
    try:
        pred_num = float(predicted_answer) if predicted_answer else 0
        truth_num = float(ground_truth)
        return abs(pred_num - truth_num) < 1e-6
    except ValueError:
        return predicted_answer.strip() == ground_truth.strip()
