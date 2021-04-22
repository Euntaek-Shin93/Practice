# %%
import numpy as np

N, D_in, H, D_out = 64,1000,100,10 # 배치, input , hidden, output
x = np.random.randn(N,D_in)
y = np.random.randn(N,D_out)
w1 = np.random.randn(D_in,H)
w2 = np.random.randn(H,D_out)

lr = 1e-6
h = x.dot(w1)  #matmul
print(h.shape)
print(x.shape)
print(w1.shape)

# %%
a = np.array([[1,2],[3,4]])
b = np.array([[1,2],[3,4]])

# %%
print(np.matmul(a,b))
print(a.dot(b))
print(np.square(a))

# %%
for t in range(500):
    h = x.dot(w1)
    h_relu = np.maximum(h,0)
    y_pred = h_relu.dot(w2)
    
    loss = np.square(y_pred - y).sum()  # L2-norm
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2

# %%
import torch

dtype = torch.float
N, D_in, H, D_out = 64,1000,100,10
x = torch.randn(N, D_in,  dtype=dtype)
y = torch.randn(N, D_out,  dtype=dtype)

w1 = torch.randn(D_in, H, dtype=dtype)
w2 = torch.randn(H, D_out, dtype=dtype)

lr = 1e-6
for t in range(500):
    # 순전파 단계: 예측값 y를 계산합니다.
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

# %%
"""
# In SGD, how to update the params

## If you want to apply another optimizer,

## you just modify 3rd line.

with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
 
This is just a SGD
If you want to apply another optimizer, you just modify 3rd line.


weight.data / weight.grad.data 

autograd의 추적을 피하기 위해서
"""

# %%
optimzer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Reference : https://tutorials.pytorch.kr/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
