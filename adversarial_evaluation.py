from evaluators.adv_evaluator import AdversarialEvaluator

if __name__ == "__main__":
    eval = AdversarialEvaluator("evaluators/config_example_date.json")
    print(eval)