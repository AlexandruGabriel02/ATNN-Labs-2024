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

WANDB_PROJECT_NAME = "hw3-part2-parameter-sweep"
cudnn.benchmark = True

class Utils:
    def get_default_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
    def has_key(d, key):
        return key in d.keys()

    def get_class_by_name(module, class_name):
        return getattr(module, class_name)
    
    def get_args_from_config(config):
        if Utils.has_key(config, "args"):
            return config["args"]
        return {}
    
    def get_transform_from_config(config):
        transforms = []
        for transform in config:
            transform_class = Utils.get_class_by_name(v2, transform["name"])
            transform_kwargs = Utils.get_args_from_config(transform)

            if (Utils.has_key(transform_kwargs, "dtype")):
                transform_kwargs["dtype"] = Utils.get_class_by_name(torch, transform_kwargs["dtype"])

            transforms.append(transform_class(**transform_kwargs))

        return v2.Compose(transforms)

class SimpleCachedDataset(Dataset):
    def __init__(self, dataset: Dataset, runtime_transforms: Optional[v2.Transform], cache: bool):
        if cache:
            dataset = tuple([x for x in dataset])
        self.dataset = dataset
        self.runtime_transforms = runtime_transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        entry, label = self.dataset[i]
        if self.runtime_transforms is None:
            return entry, label
        return self.runtime_transforms(entry), label
    

class Pipeline:
    def __init__(self):
        self.device = Utils.get_default_device()
        self.pin_memory = True
        self.enable_half = False
        self.scaler = None
        
        self.train_initial_transform = None
        self.train_runtime_transform = None
        self.test_transform = None

        self.train_set = None
        self.test_set = None

        self.train_loader = None
        self.test_loader = None

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.checkpoint_path = None
        self.early_stopping = False

        self.epochs = None

        self.wandb_config = None

    def train(self):
        self.model.train()
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.enable_half):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total

    @torch.inference_mode()
    def val(self):
        self.model.eval()
        correct = 0
        total = 0

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.enable_half):
                outputs = self.model(inputs)

            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total

    def start(self):
        best = 0.0
        best_epoch = 0
        epochs = list(range(self.epochs))
        BEST_CHECKPOINT_PATH = self.checkpoint_path
        no_improvement_cnt = 0

        wandb.init(project=WANDB_PROJECT_NAME, config=self.wandb_config)

        with tqdm(epochs) as tbar:
            for epoch in tbar:
                train_acc = self.train()
                val_acc = self.val()

                if self.scheduler is not None:
                    self.scheduler.step()
                if val_acc > best:
                    no_improvement_cnt = 0
                    best = val_acc
                    best_epoch = epoch

                    if self.checkpoint_path is not None:
                        checkpoint = {
                            'model_state_dict': self.model.state_dict(),
                        }
                        torch.save(checkpoint, BEST_CHECKPOINT_PATH)
                else:
                    # Early stopping
                    no_improvement_cnt += 1
                    if self.early_stopping is not None and no_improvement_cnt == self.early_stopping:
                        break
                
                wandb.log({"train_acc": train_acc, "val_acc": val_acc, "epoch": epoch})
                tbar.set_description(f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f} at epoch {best_epoch}")

        wandb.finish()


class PipelineBuilder:
    def __init__(self, config):
        self.config = config
        self.pipeline = Pipeline()

    def build_init_settings(self):
        if self.config["device"] != "auto":    
            self.pipeline.device = torch.device(self.config["device"])

        if Utils.has_key(self.config, "extra"):
            if Utils.has_key(self.config, "pin_memory"):
                self.pipeline.pin_memory = self.config["extra"]["pin_memory"]
            if Utils.has_key(self.config, "enable_half"):
                self.pipeline.pin_memory = self.config["extra"]["enable_half"]

        self.pipeline.scaler = GradScaler(self.pipeline.device, enabled=self.pipeline.enable_half)

    def build_transforms(self):
        if not Utils.has_key(self.config, "transforms"):
            return
        
        if Utils.has_key(self.config["transforms"], "train"):
             if Utils.has_key(self.config["transforms"]["train"], "initial"):
                 self.pipeline.train_initial_transform = Utils.get_transform_from_config(self.config["transforms"]["train"]["initial"])
            
             if Utils.has_key(self.config["transforms"]["train"], "runtime"):
                 self.pipeline.train_runtime_transform = Utils.get_transform_from_config(self.config["transforms"]["train"]["runtime"])

        if Utils.has_key(self.config["transforms"], "test"):
            self.pipeline.test_transform = Utils.get_transform_from_config(self.config["transforms"]["test"])

    def build_data(self):
        dataset_class = Utils.get_class_by_name(torchvision.datasets, self.config["dataset"]["name"])

        self.pipeline.train_set = dataset_class('./data', download=True, train=True, transform=self.pipeline.train_initial_transform)
        self.pipeline.test_set = dataset_class('./data', download=True, train=False, transform=self.pipeline.test_transform)
        self.pipeline.train_set = SimpleCachedDataset(self.pipeline.train_set, self.pipeline.train_runtime_transform, True)
        self.pipeline.test_set = SimpleCachedDataset(self.pipeline.test_set, None, True)

        train_batch_size = self.config["dataloader"]["train"]["batch_size"]
        test_batch_size = self.config["dataloader"]["test"]["batch_size"]
        train_dataloader_kwargs = Utils.get_args_from_config(self.config["dataloader"]["train"])
        test_dataloader_kwargs = Utils.get_args_from_config(self.config["dataloader"]["test"])
        self.pipeline.train_loader = DataLoader(self.pipeline.train_set, batch_size=train_batch_size, pin_memory=self.pipeline.pin_memory, **train_dataloader_kwargs)
        self.pipeline.test_loader = DataLoader(self.pipeline.test_set, batch_size=test_batch_size, pin_memory=self.pipeline.pin_memory, **test_dataloader_kwargs)

    def build_model(self):
        self.pipeline.epochs = self.config["epochs"]

        if self.config["model"]["source"] == "timm":
            kwargs = Utils.get_args_from_config(self.config["model"])
            self.pipeline.model = timm.create_model(self.config["model"]["name"], **kwargs)
        elif self.config["model"]["source"] == "implemented":
            model_class = Utils.get_class_by_name(implemented_models, self.config["model"]["name"])
            self.pipeline.model = model_class()
        else:
            raise Exception("Invalid model source")

        self.pipeline.model = self.pipeline.model.to(self.pipeline.device)
        self.pipeline.model = torch.jit.script(self.pipeline.model)

        loss_class = Utils.get_class_by_name(nn, self.config["loss"])
        self.pipeline.criterion = loss_class()

        optimizer_class = Utils.get_class_by_name(optim, self.config["optimizer"]["name"])
        optimizer_kwargs = Utils.get_args_from_config(self.config["optimizer"])
        self.pipeline.optimizer = optimizer_class(self.pipeline.model.parameters(), **optimizer_kwargs)

        if Utils.has_key(self.config, "scheduler"):
            scheduler_class = Utils.get_class_by_name(optim.lr_scheduler, self.config["scheduler"]["name"])
            scheduler_kwargs = Utils.get_args_from_config(self.config["scheduler"])
            self.pipeline.scheduler = scheduler_class(self.pipeline.optimizer, **scheduler_kwargs)

        if Utils.has_key(self.config, "checkpoint_path"):
            self.pipeline.checkpoint_path = self.config["checkpoint_path"]

        if Utils.has_key(self.config, "early_stopping"):
            self.pipeline.early_stopping = self.config["early_stopping"]

    def build_wandb_config(self):
        self.pipeline.wandb_config = {
            "epochs": self.config["epochs"],
            "model": self.config["model"]["name"],
            "data_augmentations": self.config["transforms"] if Utils.has_key(self.config, "transforms") else None,
            "optimizer": self.config["optimizer"]["name"],
            "optimizer_params": self.config["optimizer"].get("args", {}),
            "batch_size_train": self.config["dataloader"]["train"]["batch_size"],
        }

    def get_pipeline(self):
        return self.pipeline
    

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config_file.json>")
        exit(-1)

    config = None
    with open(sys.argv[1], "r") as f:
        config = json.load(f)
    if config is None:
        print("Error opening config json")
        exit(-1)
    
    builder = PipelineBuilder(config)
    builder.build_init_settings()
    builder.build_transforms()
    builder.build_data()
    builder.build_model()
    builder.build_wandb_config()
    pipeline = builder.get_pipeline()

    pipeline.start()