import torch
import torch.nn as nn

class CNNblock(nn.Module):
    def __init__(self,in_channels,out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size= (3,3),padding= 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )   
        
    def forward(self,x):
        y = self.layers(x)
        return y

class VGGNET(nn.Module):
    def __init__(self,output_size):
        self.output_size = output_size
        super().__init__()

        self.block = nn.Sequential(
            CNNblock(1,32),  #(14,14)
            CNNblock(32,64), #(7,7) 
            CNNblock(64,128),  #(4,4)
            CNNblock(128,256), #(2,2)
            CNNblock(256,512),  #(1,1)
        )
        self.layer = nn.Sequential(
            nn.Linear(512,50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50,self.output_size),
            nn.LogSoftmax(dim = -1),
        )
    
    def forward(self,x):
        assert x.dim() > 2

        if x.dim() ==3:
            x = x.view(-1,1, x.size(-2),x.size(-1))
        z = self.block(x)
        
        y = self.layer(z.squeeze())

        return y
