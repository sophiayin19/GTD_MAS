from typing import List, Dict, Any
import re

def svamp_data_process(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed_dataset = []
    for record in dataset:
        task = f"{record['Body']} {record['Question']}"
        processed_dataset.append({
            "task": task,
            "answer": str(record["Answer"])
        })
    return processed_dataset

def svamp_get_predict(model_response: str) -> str:
    """
    Extracts the final numerical answer from the model's response.
    It looks for the last number in the response.
    """
    numbers = re.findall(r'-?\d+\.?\d*', model_response)
    if numbers:
        return numbers[-1]  # Return the last number found
    return ""

def svamp_check_correctness(predicted_answer: str, ground_truth: str) -> bool:
    """
    Checks if the predicted answer matches the ground truth for SVAMP dataset.
    """
    try:
        pred_num = float(predicted_answer) if predicted_answer else 0
        truth_num = float(ground_truth)
        return abs(pred_num - truth_num) < 1e-6
    except ValueError:
        return predicted_answer.strip() == ground_truth.strip()
