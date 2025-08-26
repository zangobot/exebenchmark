from evaluators.evaluator import Evaluator


if __name__ == "__main__":
    # Example usage
    config_path = (
        "evaluators/config_example_date.json"  # Replace with your actual config path
    )
    evaluator = Evaluator(config_path)
    evaluator.evaluate(
        batch_size=1
    )
