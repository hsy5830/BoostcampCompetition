
import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from torch import nn


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        # # hidden state vector 이용
        # self.m1 = transformers.AutoModel.from_pretrained(
        #     pretrained_model_name_or_path=model_name
        # )
        ## Layer 1
        # self.linear = nn.Linear(in_features = 768, out_features = 384)
        # self.outlinear = nn.Linear(in_features = 512*384, out_features = 1)

        # ## Layer 2 - base
        # self.linear1 = nn.Linear(in_features = 768, out_features = 512)
        # self.linear2 = nn.Linear(in_features = 512, out_features = 256)
        # self.outlinear = nn.Linear(in_features = 512*256, out_features = 1)

        # ## Layer 2 - large
        # # torch.Size([16, 512, 1024])
        # self.linear1 = nn.Linear(in_features = 1024, out_features = 1024)
        # self.linear2 = nn.Linear(in_features = 1024, out_features = 512)
        # self.linear3 = nn.Linear(in_features = 512, out_features = 256)
        # self.outlinear = nn.Linear(in_features = 512*256, out_features = 1)

        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss()
        self.loss_func2 = torch.nn.HuberLoss()   

    def forward(self, x):
        # original
        x = self.plm(x)['logits']

        ### with last hidden state vector
        ## Layer 1
        # x = self.m1(x)['last_hidden_state']     # [16 * 512 * 768]
        # x = nn.Dropout(0.2)(x)
        # x = self.linear(x) # (x.shape[2], int(x.shape[2]/2))
        # x = torch.tanh(x)

        # x = x.view(x.shape[0],-1)
        # x = nn.Dropout(0.2)(x)
        # x = self.outlinear(x)

        # ## Layer 2
        # x = self.m1(x)['last_hidden_state']     # [16 * 512 * 768]
        # print('---'*20)
        # print(x.shape)
        # x = nn.Dropout(0.2)(x)
        # x = self.linear1(x)
        # x = torch.tanh(x)

        # x = nn.Dropout(0.2)(x)
        # x = self.linear2(x)
        # x = torch.tanh(x)

        # # for roberta-large
        # x = nn.Dropout(0.2)(x)
        # x = self.linear3(x)
        # x = torch.tanh(x)

        # x = x.view(x.shape[0],-1)
        # x = nn.Dropout(0.2)(x)
        # x = self.outlinear(x)
            
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())+ self.loss_func2(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())+ self.loss_func2(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=self.lr*0.001)
        return [optimizer], [lr_scheduler]
    
class KfoldModel(Model):
    def __init__(self, model_name, lr):
        super().__init__(model_name=model_name, lr=lr)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("k_test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        
class HuberModel(Model):
    def __init__(self, model_name, lr):
        super().__init__(model_name=model_name, lr=lr)
        
        self.loss_func2 = torch.nn.HuberLoss()        

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float()) + self.loss_func2(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float()) + self.loss_func2(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss