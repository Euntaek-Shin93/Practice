# %%
import torch

# %%
print(torch.tensor([1,2,3,4]))

# %%
x = torch.empty(5,3)
print(x)

# %%
y = torch.rand(5,3)

# %%
y

# %%
x = x.new_ones(5,3)
print(x)

# %%
x.size()

# %%
import numpy as np
d = np.array([[0,1,2,3,4],[5,6,7,8,9]])
print(d[0:2,1])
#numpy 에서는 tuple의 형태로 indexing 

# %%
x = torch.randn(4,4) # torch.randn -> Gaussian distribution
                     # torch.rand -> uniform distribution on [0,1]
y = x.view(16)
z = x.view(-1,8)
print(x.size(),y.size(),z.size())
print(x.shape)
print(np.shape(x))
print(x.size)

# %%
a = torch.ones(5)
b = a.numpy()
print(b)

# %%
b+=1
print(a,b)  # a와 b는 메모리공간을 공유


# %%
x = torch.ones(5,3,requires_grad= True)
y = torch.ones(5,3)

# %%
z = x+y
print(z)

# %%
a = z*z*3
out = a.mean()
print(a,out)

# %%
out = z.mean()
out.backward()

# %%
print(x.grad)
print(z.grad)
print(out)

# %%
x = torch.ones(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

# %%
v = torch.tensor([1.2,1.1,1.1], dtype=torch.float)
x.backward(v) #backward는 자세히 말하면 Jacobian matrix 와 v의 곱을 나타내는 것이다.

print(x.grad) # x.grad는 결국 Jacobian matrix * v의 값


# %%
import torch.nn as nn
import torch.nn.functional as F

# %%
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()  #nn.Module을 상속받는데 파생class에 대한 명시를 위해 Net과 self를 넣어줌
        self.conv1 = nn.Conv2d(1,6,3) # input 1 , ouput 6, kernel 3x3 
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16*6*6,120)
        self.fc2= nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self,x):
        print(x.shape,"input")
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        print(x.shape,"conv1+maxpool")
        x = F.relu(self.conv2(x))
        print(x.shape,"conv2")
        x= F.max_pool2d(x,(2,2))
        print(x.shape,"maxpool2")
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        print(x.shape,"fc1")
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        return x
        
    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features =1
        for s in size:
            num_features*=s
        return num_features
    
net = Net()
print(net)

# %%
params = list(net.parameters())  #each kernel has bias parameter
print(len(params))
for i in range(10):
    print(params[i].size())  

# %%
input = torch.randn(1,1,32,32)

# %%
out = net(input)
print(out)

# %%
out.shape

# %%
net.zero_grad()
out.backward(torch.randn(1,10))

# %%
output = net(input)
target = torch.randn(10)
target = target.view(1,-1)
criterion = nn.MSELoss()
loss = criterion(output,target)
print(loss)

# %%
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions)
print(loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions[0])
print(loss.grad_fn.next_functions[0][0].next_functions[2][0].next_functions[0])

# %%
learning_rate = 0.01

# %%
for f in net.parameters():
    print(f)

# %%
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate) 
    #f.data는 data만 뽑아주고
    #sub_ / add_ () 괄호안에 값을 더하거나 뺌
    

# %%
a = torch.tensor([1,2,3,4])
for f in a:
    print(a)
    print(a.data.add_(1))

# %%
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr = 0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output,target)
loss.backward()
optimzer.step()

# optimizer.zero_grad() -> loss.backward() -> optimzer.step()