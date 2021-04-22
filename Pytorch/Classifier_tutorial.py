# %%
import torch
import torchvision
import torchvision.transforms as transforms

# %%
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data',train = True,
                                       download=True, transform=transform)
#./data 현재 경로에 data라는 폴더에 다운로드(상대 경로)
#cd = change directory 
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,
                                         shuffle = True, num_workers = 2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    

# %%
dataiter = iter(trainloader)   # iterator 호출
images, labels = dataiter.next()
print(labels)
print(images)
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# %%
a = [1,2,3,4]
print(''.join('%2s \n' % a[i] for i in range(4)))

# %%
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)  # input 3 oupt 6 kernel 5x5
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()

# %%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# %%
for epoch in range(2):
    running_loss = 0.0
    
    for i, data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if i %2000 == 1999:
            print('[%d, %5d] loss:%.3f' % (epoch+1, i+1,running_loss/2000))
            running_loss =0.0
            
print('Finished Training')

# %%
a = [1,2,3,4]

print('%d, %d, %d' % (a[0], a[1], a[2]))

# %%
PATH = './cifar_net.pth'  #현재 경로에 저장
torch.save(net.state_dict(),PATH)
outputs = net(images)
print(outputs.size())

# %%
value, idx = torch.max(outputs,1)

print('Predicted :', ''.join(' %5s' % classes[idx[j]] for j in range(4)))

# %%
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()  #evaluation mode로 들어가 train mode 대신 dropout비활성화 / BN 저장된거 사용
