from evaluators.evaluator import Evaluator


if __name__ == "__main__":
    # Example usage
    config_path = "evaluators/config_example_date.json"  # Replace with your actual config path
    evaluator = Evaluator(config_path)
    # data_loader = evaluator.load_data()  # to load a Dataloader for the dataset indicated in the config
    evaluator.evaluate()  # to evaluate the dataset without explicitly load it
