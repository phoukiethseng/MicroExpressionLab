import os
import torch

# All runtime configuration value that this entire project depend on.
# All configuration value can be overwritten by setting the corresponding environment variable

def get_config_numerical_value(config_name, default):
    return int(get_config_str_value(config_name, default))

def get_config_str_value(config_name, default):
    return os.environ.get(config_name, default)

# Default to use to CUDA GPU compute if possible
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH = get_config_numerical_value("EPOCH", 40)
BATCH_SIZE = get_config_numerical_value("BATCH_SIZE", 128)
LEARNING_RATE = get_config_numerical_value("LEARNING_RATE", 0.001)
EXP_CLASS_SIZE = get_config_numerical_value("EXP_CLASS_SIZE", 7)
EXP_STATE_SIZE = get_config_numerical_value("EXP_STATE_SIZE", 5)

# Number of worker used to load dataset from disk, recommended default to CPU core count
WORKER_SIZE = get_config_numerical_value("WORKER_SIZE", 6)

# Location to dataset and its label files
DATASET_PATH_PREFIX = get_config_str_value("DATASET_PATH_PREFIX", "D:\\CASME2")

# Location to save trained model params and metrics
OUTPUT_PATH_PREFIX = get_config_str_value("OUTPUT_PATH_PREFIX",
                                    "D:\\CASME2_OUTPUT")
