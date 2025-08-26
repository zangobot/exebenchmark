from pathlib import Path
from maltorch.data_processing.rs_preprocessing import RandomizedAblationPreprocessing
from maltorch.data_processing.rsdel_preprocessing import RandomizedDeletionPreprocessing
from maltorch.data_processing.dynamic_random_drs_preprocessing import (
    DynamicRandomDeRandomizedPreprocessing,
)
from maltorch.data_processing.fixed_size_chunk_drs_preprocessing import (
    FixedSizeChunkDeRandomizedPreprocessing,
)
from maltorch.data_processing.k_partition_drs_preprocessing import (
    KPartitionDeRandomizedPreprocessing,
)
from maltorch.data_processing.grayscale_preprocessing import GrayscalePreprocessing

ZOO_PATH = Path(__file__).parent / "models"

#insert here path to goodware for adv attacks
BENIGNWARE_PATH = Path(__file__).parent / "goodware"

#insert here path to malware for adv attacks
MALWARE_FOR_ADV = Path(__file__).parent / "malware"


PREPROCESSING_LIST = [
            RandomizedAblationPreprocessing,
            RandomizedDeletionPreprocessing,
            DynamicRandomDeRandomizedPreprocessing,
            FixedSizeChunkDeRandomizedPreprocessing,
            KPartitionDeRandomizedPreprocessing,
            GrayscalePreprocessing,
]