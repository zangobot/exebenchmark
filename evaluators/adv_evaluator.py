from evaluators.evaluator import Evaluator


class AdversarialEvaluator(Evaluator):

    def __init__(self, config_path):
        super().__init__(config_path)

        if "attack" not in self.config:
            raise ValueError("Attack configuration must contain an 'attack' key.")

        self.attack_engine = self.create_attack()

    def create_attack(self):

        if self.config["attack"] == "gamma":
            self.attack_engine = GAMMASectionInjection(
                query_budget=20,
                benignware_folder=folder / "benignware",
                which_sections=[".rdata"],
                how_many_sections=50
            )

    def bulk_attack(self):
        pass




