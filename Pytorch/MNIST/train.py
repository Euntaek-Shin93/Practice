import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class Trainer():
    def __init__(self,model,optimizer, criterion,config,train_data_loader, val_data_loader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        super().__init__()
        
    def _train(self,train_data_loader):
        self.model.train()
        total_loss = 0
        for i, (x,y) in enumerate(train_data_loader):
            y_hat = self.model(x)
            loss = self.criterion(y_hat,y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss)
        
        return total_loss / len(train_data_loader)
    def _val(self,val_data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i,(x,y) in enumerate(val_data_loader):
                y_hat = self.model(x)
                loss = self.criterion(y_hat,y)
                total_loss+= float(loss)
        return total_loss/ len(val_data_loader)

    def train(self,train_data_loader, val_data_loader, config):
        lowest_loss = np.inf
        best_model = None
        for epoch in range(config.epochs):
            train_loss = self._train(train_data_loader)
            val_loss = self._val(val_data_loader)

            if lowest_loss >= val_loss:
                lowest_loss = val_loss
                best_model = deepcopy(self.model.state_dict())
            
            print("Epoch [{0:4d}|{1:4d}] train loss : {2:.4f} , val loss : {3:.4f}, lowest loss : {4:.4f}".format(epoch+1,config.epochs,train_loss,val_loss,lowest_loss))

        self.model.load_state_dict(best_model)
        
        

            