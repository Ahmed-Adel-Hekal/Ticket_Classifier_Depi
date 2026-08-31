import pandas as pd

from prompts.classifier import build_ticket_classifier
from structured.schemas import TicketOutput
from structured.parser import parse_and_validate_ticket

from .models import EvaluationResult


class TicketEvaluator:
    """
    Evaluates the AI Support Assistant on a ticket dataset.

    The evaluator is responsible for:
    - building the classifier prompt
    - calling the generator
    - parsing and validating the model output
    - comparing predictions with ground truth
    - preserving detailed error information
    - producing EvaluationResult objects

    It is NOT responsible for calculating aggregate metrics.
    """

    def __init__(self, generator):
        self.generator = generator

    def evaluate_ticket(self, row) -> EvaluationResult:
        """
        Evaluate a single ticket.
        """

        # ---------------------------------------------------------
        # 1. Build prompt
        # ---------------------------------------------------------
        try:
            prompt = build_ticket_classifier(
                row["message"]
            )

        except Exception as e:
            return EvaluationResult(
                ticket_id=row["ticket_id"],
                success=False,

                expected_category=row["category"],
                expected_sentiment=row["sentiment"],
                expected_urgency=row["urgency"],

                result=None,

                error=f"Prompt generation failed: {type(e).__name__}: {e}",
            )

        # ---------------------------------------------------------
        # 2. Generate model output
        # ---------------------------------------------------------
        try:
            raw_result = self.generator.generate(
                prompt,
                TicketOutput,
            )

            print(
                f"[{row['ticket_id']}] "
                f"RAW RESULT: {raw_result}"
            )

        except Exception as e:
            return EvaluationResult(
                ticket_id=row["ticket_id"],
                success=False,

                expected_category=row["category"],
                expected_sentiment=row["sentiment"],
                expected_urgency=row["urgency"],

                result=None,

                error=(
                    f"Generation failed: "
                    f"{type(e).__name__}: {e}"
                ),
            )

        # ---------------------------------------------------------
        # 3. Parse and validate output
        # ---------------------------------------------------------
        try:
            parsed_result = parse_and_validate_ticket(
                raw_result
            )

            print(
                f"[{row['ticket_id']}] "
                f"PARSED RESULT: {parsed_result}"
            )

        except Exception as e:
            return EvaluationResult(
                ticket_id=row["ticket_id"],
                success=False,

                expected_category=row["category"],
                expected_sentiment=row["sentiment"],
                expected_urgency=row["urgency"],

                result=None,

                error=(
                    f"Parsing/validation failed: "
                    f"{type(e).__name__}: {e}"
                ),
            )

        # ---------------------------------------------------------
        # 4. Compare prediction against ground truth
        # ---------------------------------------------------------
        return EvaluationResult(
            ticket_id=row["ticket_id"],
            success=True,

            expected_category=row["category"],
            expected_sentiment=row["sentiment"],
            expected_urgency=row["urgency"],

            result=parsed_result,

            category_correct=(
                parsed_result.category
                == row["category"]
            ),

            sentiment_correct=(
                parsed_result.sentiment
                == row["sentiment"]
            ),

            urgency_correct=(
                parsed_result.urgency
                == row["urgency"]
            ),

            error=None,
        )

    def evaluate(self, dataframe) -> list[EvaluationResult]:
        """
        Evaluate every ticket.

        Returns
        -------
        list[EvaluationResult]
        """

        results = []

        for _, row in dataframe.iterrows():
            result = self.evaluate_ticket(row)
            results.append(result)

        return results

    def evaluate_dataframe(self, dataframe) -> pd.DataFrame:
        """
        Evaluate the dataset and return a detailed DataFrame.

        Each row contains:
        - original ticket information
        - expected values
        - predicted values
        - correctness flags
        - success status
        - detailed error information
        """

        results = self.evaluate(dataframe)

        rows = []

        for result in results:

            # -----------------------------------------------------
            # Default prediction values
            # -----------------------------------------------------
            predicted_category = None
            predicted_sentiment = None
            predicted_urgency = None

            # -----------------------------------------------------
            # Extract prediction if generation/parsing succeeded
            # -----------------------------------------------------
            if result.success and result.result is not None:
                predicted_category = result.result.category
                predicted_sentiment = result.result.sentiment
                predicted_urgency = result.result.urgency

            # -----------------------------------------------------
            # Build detailed evaluation row
            # -----------------------------------------------------
            rows.append({
                "ticket_id": result.ticket_id,

                # Expected / ground truth
                "expected_category": result.expected_category,
                "expected_sentiment": result.expected_sentiment,
                "expected_urgency": result.expected_urgency,

                # Model prediction
                "predicted_category": predicted_category,
                "predicted_sentiment": predicted_sentiment,
                "predicted_urgency": predicted_urgency,

                # Correctness
                "category_correct": (
                    result.category_correct
                    if result.success
                    else None
                ),

                "sentiment_correct": (
                    result.sentiment_correct
                    if result.success
                    else None
                ),

                "urgency_correct": (
                    result.urgency_correct
                    if result.success
                    else None
                ),

                # Overall status
                "success": result.success,

                # Error details
                "error": result.error,
            })

        return pd.DataFrame(rows)