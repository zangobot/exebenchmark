import maltorch.adv.evasion.partialdos
from torch.utils.data import DataLoader

from evaluators.evaluator import Evaluator
from maltorch.adv.evasion.partialdos import PartialDOS

ATTACKS = {
    'Gradient' : {
        'dos' : PartialDOS
    },
    'Gradient-free':{

    }
}


class AdversarialEvaluator(Evaluator):

    def evaluate(self, data_loader: DataLoader) -> DataLoader:
        pass

    def __init__(self, config_path):
        super().__init__(config_path)

    def load_attack(self, config):
        ...