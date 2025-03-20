import lightning
import torch.nn as nn
from torch import optim

#lightning Model
def accuracy(pred, target):
    return (pred.argmax(1) == target).sum().item()
class LightningModuleModel(lightning.LightningModule):
    def __init__(self,model):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        # x=x.cuda()
        # y=y.cuda()
        pred = self.model(x)
        loss = self.criterion(pred,y)
        acc = accuracy(pred,y)
        self.logger.log_metrics({"loss": loss, "acc": acc},step=self.global_step)
        return loss
    
    def validation_step(self, val_batch,batch_idx):
        x,y = val_batch
        # x=x.cuda()
        # y=y.cuda()
        pred = self.model(x)
        loss = self.criterion(pred,y)
        acc = accuracy(pred,y)
        self.log("valmetric", acc)
        self.logger.log_metrics({"val_loss": loss, "val_acc": acc},step=batch_idx)
        # return acc
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters())