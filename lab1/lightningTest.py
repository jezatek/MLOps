# import numpy as np
# import lightning.data
from lightning.pytorch.loggers import MLFlowLogger
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torchvision.transforms as T
import models
import optuna

import models.lightningModel
import models.simpleModel

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Torch nrThreads: {torch.get_num_threads()}")

#mnist download
rootPath = "/home/jezatek/Documents/Studia/MLOps"

trainData = datasets.FashionMNIST(
    root=rootPath + "/data",
    train=True,
    download=True,
    transform=ToTensor()
)

testData = datasets.FashionMNIST(
    root=rootPath + "/data",
    train=False,
    download=True,
    transform=ToTensor()
)
# trainLoader = DataLoader(trainData, batch_size=64, shuffle=True)
# testLoader = DataLoader(testData, batch_size=64)

#Dataloaders - parallel, noise
class NoiseAndShiftDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = T.Compose([
            T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.0))
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        org = self.data[idx][0]

        # noise = self.transform(org)

        # noise = torch.randn_like(noise) * 0.05 + noise
        # noise = torch.clamp(noise, 0., 1.)
        
        return org, self.data[idx][1]
    


trainLoaderParallel = DataLoader(NoiseAndShiftDataset(trainData), batch_size=100, shuffle=True, num_workers=12)
testLoaderParallel = DataLoader(NoiseAndShiftDataset(testData), batch_size=100, num_workers=12)


checkpointCallback = ModelCheckpoint(
    monitor = "valmetric",
    mode = "max"
)
def objective(trial):
    mlfLogger = MLFlowLogger(experiment_name="FashionTest2",tracking_uri=rootPath+"/mlruns",run_name=f"Optuna run nr {trial.number}")
    convSize = trial.suggest_int('convSize',10,20)
    model = models.simpleModel.SimpleModel(10,convSize)

    torchModel = models.lightningModel.LightningModuleModel(model)

    trainer = lightning.Trainer(max_epochs=3,logger=mlfLogger,default_root_dir=rootPath+"/modelCheckpoint",callbacks=checkpointCallback)
    # trainer = lightning.Trainer(max_epochs=3,logger=mlfLogger,callbacks=checkpointCallback)
    trainer.fit(torchModel,trainLoaderParallel,testLoaderParallel)

    return trainer.callback_metrics["valmetric"].item()

study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=3)
for i,t in enumerate(study.trials):
    print(i,t.value,t.params)