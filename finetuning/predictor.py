"""
Inference/prediction for mental health classification
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Union

from .model_loader import MentalHealthModelLoader
from .config import CATEGORIES, ID2LABEL

logger = logging.getLogger(__name__)


class MentalHealthPredictor:
    """Predictor for mental health classification"""

    def __init__(self, model_path: Path, device: str = None):
        """
        Initialize predictor

        Args:
            model_path: Path to fine-tuned model
            device: Device to use (cuda/cpu), auto-detected if None
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {self.device}")

        # Load model
        self.model_loader = MentalHealthModelLoader.load_finetuned_model(model_path)
        self.model = self.model_loader.model.to(self.device)
        self.tokenizer = self.model_loader.tokenizer

        self.model.eval()
        logger.info("Model loaded and ready for inference")

    @torch.no_grad()
    def predict(
        self,
        texts: Union[str, List[str]],
        threshold: float = 0.5,
        return_probabilities: bool = False
    ) -> Union[Dict, List[Dict]]:
        """
        Predict mental health categories for input text(s)

        Args:
            texts: Single text or list of texts
            threshold: Probability threshold for positive prediction
            return_probabilities: Whether to return probabilities

        Returns:
            Predictions dictionary or list of dictionaries
        """
        # Handle single text
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits).cpu().numpy()

        # Apply threshold to get binary predictions
        predictions = (probabilities > threshold).astype(int)

        # Format results
        results = []
        for i in range(len(texts)):
            result = {
                "text": texts[i],
                "predictions": {},
                "predicted_categories": []
            }

            for category_idx, category_name in ID2LABEL.items():
                is_positive = bool(predictions[i][category_idx])
                prob = float(probabilities[i][category_idx])

                result["predictions"][category_name] = {
                    "positive": is_positive,
                    "probability": prob
                }

                if is_positive:
                    result["predicted_categories"].append(category_name)

            if return_probabilities:
                result["all_probabilities"] = {
                    cat: float(probabilities[i][idx])
                    for idx, cat in ID2LABEL.items()
                }

            results.append(result)

        # Return single result if single input
        if single_input:
            return results[0]

        return results

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        threshold: float = 0.5,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Predict for a batch of texts

        Args:
            texts: List of texts
            batch_size: Batch size for processing
            threshold: Probability threshold
            show_progress: Whether to show progress

        Returns:
            List of prediction dictionaries
        """
        all_results = []

        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            if show_progress:
                batch_num = i // batch_size + 1
                logger.info(f"Processing batch {batch_num}/{num_batches}")

            batch_results = self.predict(
                batch_texts,
                threshold=threshold,
                return_probabilities=True
            )

            all_results.extend(batch_results)

        return all_results

    def explain_prediction(
        self,
        text: str,
        threshold: float = 0.5
    ) -> Dict:
        """
        Get detailed explanation of prediction

        Args:
            text: Input text
            threshold: Probability threshold

        Returns:
            Detailed prediction explanation
        """
        result = self.predict(text, threshold=threshold, return_probabilities=True)

        explanation = {
            "text": text,
            "summary": f"Predicted {len(result['predicted_categories'])} categories",
            "categories": []
        }

        # Sort by probability
        sorted_categories = sorted(
            result["all_probabilities"].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for category, probability in sorted_categories:
            is_predicted = category in result["predicted_categories"]

            category_info = {
                "category": category,
                "probability": probability,
                "predicted": is_predicted,
                "confidence": "high" if probability > 0.8 or probability < 0.2 else
                             "medium" if probability > 0.6 or probability < 0.4 else
                             "low"
            }

            explanation["categories"].append(category_info)

        return explanation

    def get_top_k_predictions(
        self,
        text: str,
        k: int = 2
    ) -> List[Dict]:
        """
        Get top-k category predictions

        Args:
            text: Input text
            k: Number of top predictions to return

        Returns:
            List of top-k predictions
        """
        result = self.predict(text, threshold=0.0, return_probabilities=True)

        # Sort by probability
        sorted_predictions = sorted(
            result["all_probabilities"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        top_k = [
            {
                "category": category,
                "probability": probability,
                "rank": idx + 1
            }
            for idx, (category, probability) in enumerate(sorted_predictions)
        ]

        return top_k


def format_prediction_output(prediction: Dict) -> str:
    """
    Format prediction for display

    Args:
        prediction: Prediction dictionary

    Returns:
        Formatted string
    """
    output = []
    output.append("=" * 60)
    output.append("PREDICTION RESULT")
    output.append("=" * 60)
    output.append(f"\nText: {prediction['text'][:100]}...")
    output.append(f"\nPredicted Categories: {', '.join(prediction['predicted_categories']) or 'None'}")
    output.append("\n" + "-" * 60)
    output.append("Per-Category Predictions:")
    output.append("-" * 60)

    for category, pred_info in prediction["predictions"].items():
        status = "✓" if pred_info["positive"] else "✗"
        prob = pred_info["probability"]
        output.append(f"  {status} {category:<15} {prob:.4f}")

    output.append("=" * 60)

    return "\n".join(output)
