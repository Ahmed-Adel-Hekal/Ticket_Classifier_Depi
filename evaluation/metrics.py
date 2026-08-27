def accuracy (expected, predicted) : 
    if len(expected) != len(predicted) : 
        raise ValueError("Excepected and predicted must have same length")

    if len(expected) == 0 : 
        raise ValueError(
            "Cannot calculate accuracy on empty data."
        )

    correct = sum(
        expected_value == predicted_value
        for expected_value, predicted_value in 
        zip(expected, predicted)
    )

    return correct / len(predicted)