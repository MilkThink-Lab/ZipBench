"""Numeric accuracy evaluator with tolerance for floating point comparisons."""
from typing import List, Optional
import re

from opencompass.registry import ICL_EVALUATORS
from .icl_base_evaluator import BaseEvaluator


@ICL_EVALUATORS.register_module()
class NumericAccEvaluator(BaseEvaluator):
    """Accuracy evaluator with numeric tolerance for floating point comparisons.
    
    This evaluator converts predictions and references to numeric values and 
    compares them with a specified tolerance (relative or absolute error).
    
    Args:
        rtol (float): Relative tolerance. Defaults to 1e-3 (0.1%).
        atol (float): Absolute tolerance. Defaults to 1e-5.
        pred_postprocessor (optional): Function or configuration for prediction
            post-processing.
    
    Example:
        Two values are considered equal if:
        abs(pred - ref) <= atol + rtol * abs(ref)
    """

    def __init__(self,
                 rtol: float = 1e-3,
                 atol: float = 1e-5,
                 pred_postprocessor: Optional[dict] = None) -> None:
        super().__init__(pred_postprocessor=pred_postprocessor)
        self.rtol = rtol
        self.atol = atol

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numeric value from text string.
        
        Args:
            text (str): Input text that may contain a number.
            
        Returns:
            float or None: Extracted numeric value, or None if extraction fails.
        """
        if text is None:
            return None
            
        text = str(text).strip()
        
        # Try direct conversion first
        try:
            return float(text)
        except ValueError:
            pass
        
        # Try to extract number using regex
        # Match patterns like: -123.456, 1.23e-5, etc.
        patterns = [
            r'-?\d+\.?\d*[eE][+-]?\d+',  # Scientific notation
            r'-?\d+\.\d+',                # Decimal number
            r'-?\d+',                     # Integer
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    continue
        
        return None

    def _is_numeric_match(self, pred: str, ref: str) -> bool:
        """Check if prediction matches reference within tolerance.
        
        Args:
            pred (str): Prediction value.
            ref (str): Reference value.
            
        Returns:
            bool: True if values match within tolerance, False otherwise.
        """
        pred_num = self._extract_number(pred)
        ref_num = self._extract_number(ref)
        
        # If either extraction fails, fall back to string comparison
        if pred_num is None or ref_num is None:
            return str(pred).strip() == str(ref).strip()
        
        # Check numeric equality with tolerance
        # Formula: abs(a - b) <= atol + rtol * abs(b)
        diff = abs(pred_num - ref_num)
        tolerance = self.atol + self.rtol * abs(ref_num)
        
        return diff <= tolerance

    def score(self, predictions: List, references: List, test_set=None) -> dict:
        """Calculate accuracy with numeric tolerance.
        
        Args:
            predictions (List): List of predictions.
            references (List): List of reference answers.
            test_set: Test dataset (not used).
            
        Returns:
            dict: Dictionary containing 'accuracy' score (0-100).
        """
        if len(predictions) != len(references):
            raise ValueError(
                f'Number of predictions ({len(predictions)}) does not match '
                f'number of references ({len(references)})'
            )
        
        correct = 0
        total = len(predictions)
        
        for pred, ref in zip(predictions, references):
            if self._is_numeric_match(pred, ref):
                correct += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        return {'accuracy': accuracy}
