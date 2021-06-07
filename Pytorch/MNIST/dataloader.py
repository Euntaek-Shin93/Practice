from torch.utils import data
import torchvision
import torch
from torch.utils.data import DataLoader, Dataset
def load_MNIST():

    MNIST_train_data = torchvision.datasets.MNIST(root = '../data', train = True, transform= torchvision.transforms.ToTensor(), download=True)
    x = MNIST_train_data.data.float()/255.
    y = MNIST_train_data.targets
    
    MNIST_test_data = torchvision.datasets.MNIST(root =  '../data', train = False, transform = torchvision.transforms.ToTensor(), download=True)
    test_x = MNIST_test_data.data.float()/255.
    test_y = MNIST_test_data.targets
    

    return x,y, test_x,test_y

class MNISTDataset(Dataset):
    def __init__(self,data,label):
        self.data = data
        self.label = label

        super().__init__()
    
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return x, y

def get_loaders(config):
    x, y, test_x,test_y = load_MNIST()
    train_cnt = int(x.size(0)*0.8)
    val_cnt = int(x.size(0)*0.2)
    indices = torch.randperm(x.size(0))
    train_x , val_x = torch.index_select(x, dim = 0 , index = indices).split([train_cnt,val_cnt])
    train_y, val_y = torch.index_select(y,dim = 0 , index = indices).split([train_cnt,val_cnt])
    
    train_data_loader = DataLoader(MNISTDataset(train_x,train_y),
                                            batch_size=config.batch_size,
                                            shuffle=True)
    val_data_loader = DataLoader(MNISTDataset(val_x,val_y), batch_size=config.batch_size,shuffle=True)
    test_data_loader = DataLoader(MNISTDataset(test_x,test_y),batch_size= config.batch_size)

    return train_data_loader,val_data_loader,test_data_loader

