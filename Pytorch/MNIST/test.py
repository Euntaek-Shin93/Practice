import torch
import torch.nn as nn
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from model import VGGNET
from dataloader import load_MNIST

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_file_name', required=True)
    p.add_argument('--batch_size',type = int, default = 32)
    p.add_argument('--gpu',type = int, default = 0 if torch.cuda.is_available() else - 1)
    config = p.parse_args()
    return config

def test(model,x,y):
    model.eval()

    with torch.no_grad():
        y_hat = model(x)
        correct = (torch.argmax(y_hat, dim = -1))==y.squeeze().sum()
        total = float(x.size(0))
        acc = correct / total
        print("Accuracy : {0:.4f}".format(acc))
def plot(x,y_hat):
    for i in range(9):
        img = (np.array(x[i].detach().cpu(), dtype='float')).reshape(28,28)
        plt.imshow(img,cmap = 'gray')
        plt.show()
        print("Predict as :", float(torch.argmax(y_hat[i], dim = -1)))
def main(config):
    device = torch.device('cpu') if config.gpu < 0 else torch.device('cuda:{:d}'.format(config.gpu))
    _,_,test_x,test_y = load_MNIST()
    model = VGGNET(10).to(device)
    checkpoint = torch.load("./my_model.pth")
    model.load_state_dict(checkpoint['model'])
    x, y = test_x,test_y
    x,y = x.to(device), y.to(device)
    test(model,x,y)
    plot(x[:9],model(x[:9]))
if __name__ == '__main__':
    config = argparser()
    main(config)
