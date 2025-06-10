import multiprocessing

from evaluators.adv_evaluator import AdversarialEvaluator

if __name__ == "__main__":
    eval = AdversarialEvaluator("evaluators/config_example_date.json")
    eval.create_attack()
    multiprocessing.set_start_method("spawn")
    eval.bulk_attack(n_jobs=2)
    print(eval)