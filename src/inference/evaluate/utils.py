"""
This module provides utility classes for performance tracking and evaluation of classification tasks:
- `ClassificationResults` to collect and summarize results across multiple windows or files.
- `TimeMemoryTracer` to monitor the time and memory footprint of individual classification calls.
"""
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import time
import tracemalloc

from typing import List, Optional, Tuple


class ClassificationResults:
    """
    An object holding aggregate classification results containing:
        actual_annotations (Optional[List[str]]): Combined list of all actual annotations.
        predicted_annotations (Optional[List[str]]): Combined list of all predicted annotations.
        max_classification_time_ms (float): The highest classification time (ms) observed.
        max_classification_memory_kb (float): The highest memory usage (kB) observed.
        total_prompt_tokens (int): The total number of tokens needed for the classification.
    """
    def __init__(self, actual_annotations: Optional[List[str]] = None, predicted_annotations: Optional[List[str]] = None,
        max_classification_time_ms: float = 0.0, max_classification_memory_kb: float = 0.0) -> None:
        if max_classification_time_ms < 0:
            raise ValueError(f'Time must be larger than 0, got {max_classification_time_ms} ms')
        if max_classification_memory_kb < 0:
            raise ValueError(f'Memory must be larger than 0, got {max_classification_memory_kb} kb.')

        self.actual_annotations = actual_annotations if actual_annotations is not None else []
        self.predicted_annotations = predicted_annotations if predicted_annotations is not None else []
        self.max_classification_time_ms = max_classification_time_ms
        self.max_classification_memory_kb = max_classification_memory_kb

    def update(self, other: 'ClassificationResults') -> None:
        """
        Merges another ClassificationResults into this one, in place.
        """
        self.actual_annotations.extend(other.actual_annotations)
        self.predicted_annotations.extend(other.predicted_annotations)
        if other.max_classification_time_ms > self.max_classification_time_ms:
            self.max_classification_time_ms = other.max_classification_time_ms
        if other.max_classification_memory_kb > self.max_classification_memory_kb:
            self.max_classification_memory_kb = other.max_classification_memory_kb

    def infer_classes(self) -> List[str]:
        """
        Infers the full set of class labels from ground-truth and predicted annotations
        from the actual and predicted annotation lists.

        Returns:
            List[str]: A sorted list of unique annotations.
        """
        unique_actual = set(self.actual_annotations)
        unique_predicted = set(self.predicted_annotations)
        all_labels = unique_actual.union(unique_predicted)
        classes = sorted(all_labels)
        return classes

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
                - 'max_classification_time_ms' (float): The worst-case classification time (ms).
                - 'peak_classification_memory_kb' (float): The worst-case memory usage (kB),
                - 'total_classification_time_secs' (float): Total classification time in seconds.
        """
        if self.actual_annotations is None:
            raise ValueError(f'Actual annotations list empty, cannot generate a report.')
        if self.predicted_annotations is None:
            raise ValueError(f'Predicted annotations list empty, cannot generate a report.')

        classes = self.infer_classes()
        c_matrix = confusion_matrix(self.actual_annotations, self.predicted_annotations, labels=classes)
        total_no_predictions = len(self.predicted_annotations)
        accuracy = accuracy_score(self.actual_annotations, self.predicted_annotations)
        weighted_avg_precision = precision_score(self.actual_annotations, self.predicted_annotations, 
                                            average='weighted', labels=classes, zero_division=0)
        weighted_avg_recall = recall_score(self.actual_annotations, self.predicted_annotations, 
                                        average='weighted', labels=classes, zero_division=0)
        weighted_avg_f1 = f1_score(self.actual_annotations, self.predicted_annotations, 
                                        average='weighted', labels=classes, zero_division=0)

        return {
            'classes': classes,
            'confusion_matrix': c_matrix.tolist(),
            'total_no_predictions': total_no_predictions,
            'accuracy': accuracy,
            'weighted_avg_precision': weighted_avg_precision,
            'weighted_avg_recall': weighted_avg_recall,
            'weighted_avg_f1': weighted_avg_f1,
            'max_classification_time_ms': self.max_classification_time_ms,
            'peak_classification_memory_kb': self.max_classification_memory_kb,
            'total_classification_time_secs': total_classification_time_secs
        }

class TimeMemoryTracer:
    """
    Tracer for timing and peak memory usage.
    """
    def __init__(self):
        tracemalloc.start()
        self._start_time = time.perf_counter()

    def _get_time(self) -> float:
        elapsed_secs = time.perf_counter() - self._start_time
        elapsed_ms = elapsed_secs * 1000
        return elapsed_ms

    def _get_memory(self) -> float:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_kb = peak_bytes / 1024
        return peak_kb
    
    def stop(self) -> Tuple[float, float]:
        """
        Stops tracing and returns elapsed time (ms) and peak memory (KB).

        Returns:
            Tuple of (elapsed_ms, peak_kb)
        """
        return self._get_time(), self._get_memory()