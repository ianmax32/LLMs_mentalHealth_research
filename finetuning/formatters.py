"""
Output formatters for different use cases
"""

from typing import Dict, List


def format_as_positive_negative(prediction: Dict) -> List[str]:
    """
    Format prediction as simple Positive/Negative labels

    Args:
        prediction: Prediction dictionary from predictor

    Returns:
        List of formatted strings like "Positive - Anxiety"

    Example:
        >>> result = predictor.predict("I feel anxious")
        >>> labels = format_as_positive_negative(result)
        >>> print(labels)
        ['Positive - Anxiety', 'Negative - Depression', 'Negative - Psychosis', 'Negative - Mania']
    """
    formatted = []

    for category, pred_info in prediction["predictions"].items():
        status = "Positive" if pred_info["positive"] else "Negative"
        formatted.append(f"{status} - {category}")

    return formatted


def format_as_csv_row(prediction: Dict) -> str:
    """
    Format prediction as CSV row

    Returns:
        CSV string: "text,Anxiety,Depression,Psychosis,Mania"

    Example:
        >>> result = predictor.predict("I feel anxious")
        >>> csv = format_as_csv_row(result)
        >>> print(csv)
        "I feel anxious",Positive,Negative,Negative,Negative
    """
    text = prediction["text"].replace('"', '""')  # Escape quotes

    labels = []
    for category in ["Anxiety", "Depression", "Psychosis", "Mania"]:
        status = "Positive" if prediction["predictions"][category]["positive"] else "Negative"
        labels.append(status)

    return f'"{text}",{",".join(labels)}'


def format_as_binary(prediction: Dict) -> Dict[str, int]:
    """
    Format prediction as binary 0/1

    Returns:
        Dictionary with 1 for positive, 0 for negative

    Example:
        >>> result = predictor.predict("I feel anxious")
        >>> binary = format_as_binary(result)
        >>> print(binary)
        {'Anxiety': 1, 'Depression': 0, 'Psychosis': 0, 'Mania': 0}
    """
    return {
        category: 1 if pred_info["positive"] else 0
        for category, pred_info in prediction["predictions"].items()
    }


def format_as_label_list(prediction: Dict) -> str:
    """
    Format as comma-separated list of positive labels only

    Returns:
        String like "Anxiety, Depression" or "None" if no positive labels

    Example:
        >>> result = predictor.predict("I feel anxious and sad")
        >>> labels = format_as_label_list(result)
        >>> print(labels)
        "Anxiety, Depression"
    """
    positive_labels = prediction["predicted_categories"]
    return ", ".join(positive_labels) if positive_labels else "None"


def format_as_table(predictions: List[Dict]) -> str:
    """
    Format multiple predictions as a table

    Args:
        predictions: List of prediction dictionaries

    Returns:
        Formatted table string
    """
    lines = []

    # Header
    lines.append("=" * 100)
    lines.append(f"{'Text':<40} {'Anxiety':<12} {'Depression':<12} {'Psychosis':<12} {'Mania':<12}")
    lines.append("-" * 100)

    # Rows
    for pred in predictions:
        text = pred["text"][:37] + "..." if len(pred["text"]) > 40 else pred["text"]

        statuses = []
        for category in ["Anxiety", "Depression", "Psychosis", "Mania"]:
            status = "✓ Positive" if pred["predictions"][category]["positive"] else "✗ Negative"
            statuses.append(status)

        lines.append(f"{text:<40} {statuses[0]:<12} {statuses[1]:<12} {statuses[2]:<12} {statuses[3]:<12}")

    lines.append("=" * 100)

    return "\n".join(lines)


def format_with_confidence(prediction: Dict, show_all: bool = False) -> List[str]:
    """
    Format with confidence levels

    Args:
        prediction: Prediction dictionary
        show_all: If True, show all categories. If False, only positive ones.

    Returns:
        List of formatted strings

    Example:
        >>> result = predictor.predict("I feel anxious")
        >>> labels = format_with_confidence(result)
        >>> print("\\n".join(labels))
        Positive - Anxiety (82% confidence)
    """
    formatted = []

    for category, pred_info in prediction["predictions"].items():
        is_positive = pred_info["positive"]

        if not show_all and not is_positive:
            continue

        status = "Positive" if is_positive else "Negative"
        prob = pred_info["probability"]
        confidence = int(prob * 100)

        formatted.append(f"{status} - {category} ({confidence}% confidence)")

    return formatted


def export_to_csv(predictions: List[Dict], output_file: str):
    """
    Export predictions to CSV file

    Args:
        predictions: List of prediction dictionaries
        output_file: Output CSV file path
    """
    import csv

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['Text', 'Anxiety', 'Depression', 'Psychosis', 'Mania',
                        'Predicted_Categories', 'All_Positive'])

        # Rows
        for pred in predictions:
            text = pred["text"]

            statuses = [
                "Positive" if pred["predictions"][cat]["positive"] else "Negative"
                for cat in ["Anxiety", "Depression", "Psychosis", "Mania"]
            ]

            predicted_categories = ", ".join(pred["predicted_categories"]) or "None"
            all_positive = "Yes" if len(pred["predicted_categories"]) == 4 else "No"

            writer.writerow([text] + statuses + [predicted_categories, all_positive])
