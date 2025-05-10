"""
This code summarizes the final results of the annotation process with FM (after stages 1 and 2). 
The purpose is to test how well an online FM annotated data through an API.
"""
from typing import List, Optional

from inference.evaluate.utils import ClassificationResults


class ResultsFM(ClassificationResults):
    """
    Extends 'ClassificationResults' class with 'total_prompt_tokens'.

    Args:
        actual_annotations (Optional[List[str]]): Combined list of all actual annotations.
        predicted_annotations (Optional[List[str]]): Combined list of all predicted annotations.
        max_classification_time_ms (float): The highest classification time (ms) observed.
        max_classification_memory_kb (float): The highest memory usage (kB) observed.
        total_prompt_tokens (int): The total number of tokens needed for the classification.
    """
    def __init__(self, actual_annotations: Optional[List[str]] = None, predicted_annotations: Optional[List[str]] = None,
        max_classification_time_ms: float = 0.0, max_classification_memory_kb: float = 0.0, total_prompt_tokens: int = 0) -> None:
        if total_prompt_tokens < 0:
            raise ValueError(f'Number of tokens must be larger than 0, got {total_prompt_tokens}')
        
        super().__init__(actual_annotations, predicted_annotations, max_classification_time_ms, max_classification_memory_kb)
        self.total_prompt_tokens: int = total_prompt_tokens

    def update(self, other: 'ResultsFM') -> None:
        """
        Merges processing information of a window into the aggregate results.
        """
        super().update(other)
        self.total_prompt_tokens += other.total_prompt_tokens

    def generate_report(self, total_classification_time_secs: float) -> dict:
        """
        Generates a performance report from the aggregated classification results.

        Args:
            total_classification_time_secs (float): Total classification time in seconds.

        Returns:
            dict: A report containing:
                - 'classes' (List[str]): A sorted list of unique true and false annotations.
                - 'confusion_matrix' (List[List[int]]): Confusion matrix between true and predicted labels.
                - 'total_no_predictions' (int): Total number of predictions made (i.e., number of windows).
                - 'accuracy' (float): Overall classification accuracy.
                - 'weighted_avg_precision' (float): Weighted average precision.
                - 'weighted_avg_recall' (float): Weighted average recall.
                - 'weighted_avg_f1_score' (float): Weighted average F1 score.
                - 'max_classification_time_s' (float): The worst-case classification time (s).
                - 'max_classification_memory_kb' (float): The worst-case memory usage (kB),
                - 'total_classification_time_secs' (float): Total classification time in seconds.
                - 'total_prompt_tokens' (int): Total number of tokens needed for the classification.
        """
        report = super().generate_report(total_classification_time_secs)
        report['max_classification_time_s'] = report.pop('max_classification_time_ms') / 1000
        report['total_prompt_tokens'] = self.total_prompt_tokens
        return report

def calculate_total_classification_time_secs(List['ResultsFM']) -> float:
    pass

def calculate_total_prompt_tokens(List['ResultsFM']) -> int:
    pass