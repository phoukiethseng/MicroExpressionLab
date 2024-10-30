from datetime import datetime

import torch
from .config import *

def current_time_as_str():
    return datetime.now().strftime('%Y-%m-%d_%H-%M')

def save_model(model, model_name):
    assert model is not None
    assert model_name is not None

    time_str= current_time_as_str()
    torch.save(model.state_dict(), os.path.join(OUTPUT_PATH_PREFIX, "trained", f"{model_name}_{time_str}.pt"))

def save_metrics(metric, metric_name, model_name):
    assert metric is not None
    assert metric_name is not None
    assert model_name is not None

    time_str= current_time_as_str()
    state_dict = metric.state_dict()
    torch.save(state_dict, os.path.join(OUTPUT_PATH_PREFIX, "metrics", f"{metric_name}_of_{model_name}_{time_str}.pt"))

