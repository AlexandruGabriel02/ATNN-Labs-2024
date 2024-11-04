import torch
import torchvision.datasets
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torch.backends import cudnn
from torch import GradScaler
from torch import optim
from tqdm import tqdm
from typing import Optional
import numpy as np
from torch.utils.data import default_collate
import sys
import json
import implemented_models
import timm
import wandb
import uuid
import subprocess


# Grid search on model, optimizer and data augmentations
if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <sweep_parameters.json> <train_config.json>")
        exit(-1)

    config = None
    sweep_params = None

    with open(sys.argv[1], "r") as f:
        sweep_params = json.load(f)
    with open(sys.argv[2], "r") as f:
        config = json.load(f)

    if config is None or sweep_params is None:
        print("Error opening json")
        exit(-1)

    cnt = 0
    for model in sweep_params["model"]:
        for optimizer in sweep_params["optimizer"]:
            for augmentation in sweep_params["transforms"]:
                config["model"] = model
                config["optimizer"] = optimizer
                config["transforms"] = augmentation

                new_config_json = f"config_{uuid.uuid4()}.json"
                with open(new_config_json, "w") as f:
                    json.dump(config, f)

                cnt += 1
                print(f"Running configuration #{cnt}")
                subprocess.run(["python", "pipeline.py", new_config_json], capture_output=True, text=True)
                