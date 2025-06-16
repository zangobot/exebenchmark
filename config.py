from pathlib import Path

ZOO_PATH = Path(__file__).parent / "models"
BENIGNWARE_PATH = Path(__file__).parent / "win_exe" / "win11" / "syswow64"
MALWARE_FOR_ADV = Path(__file__).parent / "malware"

MODEL_LIST = [
    "EmberGBDT",
    "ResNet18",
    "MalConv",
    "BBDnn",
    "AvastStyleConv",
    "NgramConv",
]
