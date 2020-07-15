import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim


x = torch.unsqueeze(torch.linspace(0,10,50),dim=1)
y = x + 2 * torch.rand(x.size())

# plt.scatter(x,y)
# plt.plot(x,y)


# x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
#                     [9.779], [6.182], [7.59], [2.167], [7.042],
#                     [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

# y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
#                     [3.366], [2.596], [2.53], [1.221], [2.827],
#                     [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


# x_train = torch.from_numpy(x_train)

# y_train = torch.from_numpy(y_train)

# x = x_train
# y = y_train



a = np.array([[1.2]])

print(x.size())
print(x)
print(y.size())
print(y)
# plt.scatter(x.numpy(),y.numpy())
# plt.show()

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        out = self.linear(x)
        return out


def main():
    model = LinearRegression()

    optimizer = optim.SGD(model.parameters(),lr=1e-4)
    creterion = nn.MSELoss()

    for step in range(1000):

        out = model(x)
        loss = creterion(out,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print("step:{} loss:{}".format(step,loss))

    model.eval()
    with torch.no_grad():
        predict = model(x)
    
    plt.scatter(x,y)
    plt.plot(x,predict.data.numpy())
    plt.show()
    

main()