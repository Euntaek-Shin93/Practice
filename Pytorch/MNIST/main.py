import argparse
from dataloader import get_loaders
from model import VGGNET
from train import Trainer
import torch
import torch.nn as nn
import torch.optim as optim

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_file_name',required=True)
    p.add_argument('--batch_size', type = int, default=32)
    p.add_argument('--epochs',type = int , default = 20)
    p.add_argument('--gpu',type = int, default = 0 if torch.cuda.is_available() else - 1)
    p.add_argument('--model', type = str, default = 'VGGNET')
    config = p.parse_args()
    return config

def model_maker(config):
    if config.model == 'VGGNET':
        model = VGGNET(10)

    return model
def main(config):
    device = torch.device('cpu') if config.gpu < 0 else torch.device('cuda:{:d}'.format(config.gpu))
    train_data_loader,val_data_loader,test_data_loader = get_loaders(config)
    model = model_maker(config).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()
    trainer = Trainer(model,optimizer, criterion,config,train_data_loader, val_data_loader)
    trainer.train(train_data_loader, val_data_loader, config)
    torch.save({'model' : trainer.model.state_dict(), 'config' : config}, config.model_file_name)


if __name__ == '__main__':
    config = argparser()
    main(config)
