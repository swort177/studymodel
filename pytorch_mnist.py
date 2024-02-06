#! /usr/bin/env python3
# -*- coding:utf-8 -*-
"""
__author__ = 'Swort'
__date__ = '2024/2/5 20:30'
mnist  pytorch gan sample
copy from https://github.com/lyeoni/pytorch-mnist-GAN/blob/master/pytorch-mnist-GAN.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bs = 100  # 批次batch_size
z_dim = 100  # 输入特征数量
lr = 0.0002  # optimizer
criterion = nn.BCELoss()  # loss 损失函数  要么是0 要么是1

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])])
# dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

# Data Loader (Input Pipeline)
train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

# 784 28*28
mnist_dim = train_dataset.data.size(1) * train_dataset.data.size(2)


class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

        # forward method

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


# print(f"{train_dataset.data.shape},{train_dataset.data.size(1)},{train_dataset.data.size(2)}")


G = Generator(g_input_dim=z_dim, g_output_dim=mnist_dim).to(device)

D = Discriminator(mnist_dim).to(device)

# 启动随机梯度下降优化算法来设置优化器 学习率为0.0002  10e4
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)


# 定义生成器的训练
def G_train(x):
    # =======================Train the generator=======================#
    G.zero_grad()
    z = torch.randn(bs, z_dim).to(device)
    y = torch.ones(bs, 1).to(device)
    # 随机生成器
    G_output = G(z)

    # 用实例化后的鉴别器去鉴别生成器的数字
    D_output = D(G_output)
    # 计算 要么是0 要么是1 损失函数
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    # 返回损失值
    return G_loss.data.item()


# 定义鉴别器的训练
def D_train(x):
    # =======================Train the discriminator=======================#
    D.zero_grad()
    # 鉴别器的训练
    # train discriminator on real
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(bs, 1)
    x_real, y_real = x_real.to(device), y_real.to(device)
    # 得到真实训练损失值
    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)

    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(bs, z_dim).to(device)
    x_fake, y_fake = G(z), torch.zeros(bs, 1).to(device)

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)

    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss

    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


n_epoch = 200
for epoch in range(1, n_epoch + 1):
    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        D_losses.append(D_train(x))
        G_losses.append(G_train(x))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
        (epoch),
        n_epoch,
        torch.mean(torch.FloatTensor(D_losses)),
        torch.mean(torch.FloatTensor(G_losses))))

    with torch.no_grad():
        test_z = torch.randn(bs, z_dim).to(device)
        generated = G(test_z)

        save_image(generated.view(generated.size(0), 1, 28, 28), './samples/sample_' + str(epoch) + '.png')


#
# [1/200]: loss_d: 0.815, loss_g: 3.484
# [2/200]: loss_d: 0.957, loss_g: 2.222
# [3/200]: loss_d: 0.885, loss_g: 2.166
# [4/200]: loss_d: 0.541, loss_g: 2.670
# [5/200]: loss_d: 0.542, loss_g: 2.768
# [6/200]: loss_d: 0.629, loss_g: 2.424
# [7/200]: loss_d: 0.598, loss_g: 2.517
# [8/200]: loss_d: 0.640, loss_g: 2.352
# [9/200]: loss_d: 0.663, loss_g: 2.280
# [10/200]: loss_d: 0.641, loss_g: 2.327
# [11/200]: loss_d: 0.688, loss_g: 2.250
# [12/200]: loss_d: 0.730, loss_g: 2.165
# [13/200]: loss_d: 0.797, loss_g: 1.942
# [14/200]: loss_d: 0.737, loss_g: 2.044
# [15/200]: loss_d: 0.767, loss_g: 1.945
# [16/200]: loss_d: 0.818, loss_g: 1.837
# [17/200]: loss_d: 0.813, loss_g: 1.842
# [18/200]: loss_d: 0.864, loss_g: 1.679
# [19/200]: loss_d: 0.852, loss_g: 1.746
# [20/200]: loss_d: 0.862, loss_g: 1.747
# [21/200]: loss_d: 0.888, loss_g: 1.647
# [22/200]: loss_d: 0.902, loss_g: 1.613
# [23/200]: loss_d: 0.939, loss_g: 1.510
# [24/200]: loss_d: 0.949, loss_g: 1.528
# [25/200]: loss_d: 0.968, loss_g: 1.480
# [26/200]: loss_d: 0.972, loss_g: 1.445
# [27/200]: loss_d: 0.987, loss_g: 1.449
# [28/200]: loss_d: 0.998, loss_g: 1.402
# [29/200]: loss_d: 0.983, loss_g: 1.450
# [30/200]: loss_d: 0.992, loss_g: 1.419
# [31/200]: loss_d: 1.016, loss_g: 1.356
# [32/200]: loss_d: 1.025, loss_g: 1.336
# [33/200]: loss_d: 1.028, loss_g: 1.351
# [34/200]: loss_d: 1.018, loss_g: 1.348
# [35/200]: loss_d: 1.053, loss_g: 1.301
# [36/200]: loss_d: 1.060, loss_g: 1.263
# [37/200]: loss_d: 1.060, loss_g: 1.272
# [38/200]: loss_d: 1.078, loss_g: 1.234
# [39/200]: loss_d: 1.085, loss_g: 1.238
# [40/200]: loss_d: 1.103, loss_g: 1.195
# [41/200]: loss_d: 1.113, loss_g: 1.176
# [42/200]: loss_d: 1.109, loss_g: 1.191
# [43/200]: loss_d: 1.096, loss_g: 1.206
# [44/200]: loss_d: 1.111, loss_g: 1.176
# [45/200]: loss_d: 1.135, loss_g: 1.147
# [46/200]: loss_d: 1.134, loss_g: 1.135
# [47/200]: loss_d: 1.153, loss_g: 1.097
# [48/200]: loss_d: 1.154, loss_g: 1.107
# [49/200]: loss_d: 1.168, loss_g: 1.072
# [50/200]: loss_d: 1.160, loss_g: 1.104
# [51/200]: loss_d: 1.156, loss_g: 1.096
# [52/200]: loss_d: 1.177, loss_g: 1.062
# [53/200]: loss_d: 1.183, loss_g: 1.050
# [54/200]: loss_d: 1.180, loss_g: 1.065
# [55/200]: loss_d: 1.171, loss_g: 1.079
# [56/200]: loss_d: 1.166, loss_g: 1.080
# [57/200]: loss_d: 1.170, loss_g: 1.077
# [58/200]: loss_d: 1.177, loss_g: 1.067
# [59/200]: loss_d: 1.167, loss_g: 1.073
# [60/200]: loss_d: 1.179, loss_g: 1.058
# [61/200]: loss_d: 1.195, loss_g: 1.039
# [62/200]: loss_d: 1.190, loss_g: 1.042
# [63/200]: loss_d: 1.184, loss_g: 1.064
# [64/200]: loss_d: 1.191, loss_g: 1.025
# [65/200]: loss_d: 1.203, loss_g: 1.009
# [66/200]: loss_d: 1.207, loss_g: 1.020
# [67/200]: loss_d: 1.201, loss_g: 1.020
# [68/200]: loss_d: 1.207, loss_g: 0.998
# [69/200]: loss_d: 1.217, loss_g: 0.996
# [70/200]: loss_d: 1.217, loss_g: 0.998
# [71/200]: loss_d: 1.213, loss_g: 1.000
# [72/200]: loss_d: 1.217, loss_g: 0.994
# [73/200]: loss_d: 1.213, loss_g: 0.991
# [74/200]: loss_d: 1.217, loss_g: 0.999
# [75/200]: loss_d: 1.219, loss_g: 0.979
# [76/200]: loss_d: 1.231, loss_g: 0.971
# [77/200]: loss_d: 1.225, loss_g: 0.997
# [78/200]: loss_d: 1.221, loss_g: 0.990
# [79/200]: loss_d: 1.223, loss_g: 0.981
# [80/200]: loss_d: 1.229, loss_g: 0.984
# [81/200]: loss_d: 1.226, loss_g: 0.973
# [82/200]: loss_d: 1.234, loss_g: 0.954
# [83/200]: loss_d: 1.235, loss_g: 0.970
# [84/200]: loss_d: 1.233, loss_g: 0.970
# [85/200]: loss_d: 1.232, loss_g: 0.971
# [86/200]: loss_d: 1.241, loss_g: 0.959
# [87/200]: loss_d: 1.237, loss_g: 0.967
# [88/200]: loss_d: 1.236, loss_g: 0.969
# [89/200]: loss_d: 1.234, loss_g: 0.966
# [90/200]: loss_d: 1.234, loss_g: 0.973
# [91/200]: loss_d: 1.237, loss_g: 0.954
# [92/200]: loss_d: 1.246, loss_g: 0.948
# [93/200]: loss_d: 1.248, loss_g: 0.947
# [94/200]: loss_d: 1.246, loss_g: 0.946
# [95/200]: loss_d: 1.244, loss_g: 0.956
# [96/200]: loss_d: 1.241, loss_g: 0.947
# [97/200]: loss_d: 1.248, loss_g: 0.951
# [98/200]: loss_d: 1.249, loss_g: 0.936
# [99/200]: loss_d: 1.246, loss_g: 0.954
# [100/200]: loss_d: 1.240, loss_g: 0.961
# [101/200]: loss_d: 1.249, loss_g: 0.936
# [102/200]: loss_d: 1.250, loss_g: 0.942
# [103/200]: loss_d: 1.256, loss_g: 0.934
# [104/200]: loss_d: 1.253, loss_g: 0.932
# [105/200]: loss_d: 1.259, loss_g: 0.923
# [106/200]: loss_d: 1.251, loss_g: 0.938
# [107/200]: loss_d: 1.253, loss_g: 0.934
# [108/200]: loss_d: 1.254, loss_g: 0.928
# [109/200]: loss_d: 1.257, loss_g: 0.932
# [110/200]: loss_d: 1.260, loss_g: 0.919
# [111/200]: loss_d: 1.256, loss_g: 0.935
# [112/200]: loss_d: 1.251, loss_g: 0.951
# [113/200]: loss_d: 1.256, loss_g: 0.927
# [114/200]: loss_d: 1.266, loss_g: 0.913
# [115/200]: loss_d: 1.254, loss_g: 0.929
# [116/200]: loss_d: 1.266, loss_g: 0.913
# [117/200]: loss_d: 1.268, loss_g: 0.921
# [118/200]: loss_d: 1.266, loss_g: 0.917
# [119/200]: loss_d: 1.260, loss_g: 0.919
# [120/200]: loss_d: 1.268, loss_g: 0.902
# [121/200]: loss_d: 1.264, loss_g: 0.914
# [122/200]: loss_d: 1.270, loss_g: 0.913
# [123/200]: loss_d: 1.266, loss_g: 0.915
# [124/200]: loss_d: 1.268, loss_g: 0.904
# [125/200]: loss_d: 1.271, loss_g: 0.912
# [126/200]: loss_d: 1.265, loss_g: 0.915
# [127/200]: loss_d: 1.268, loss_g: 0.910
# [128/200]: loss_d: 1.266, loss_g: 0.909
# [129/200]: loss_d: 1.264, loss_g: 0.916
# [130/200]: loss_d: 1.264, loss_g: 0.922
# [131/200]: loss_d: 1.271, loss_g: 0.897
# [132/200]: loss_d: 1.272, loss_g: 0.909
# [133/200]: loss_d: 1.274, loss_g: 0.897
# [134/200]: loss_d: 1.276, loss_g: 0.902
# [135/200]: loss_d: 1.270, loss_g: 0.916
# [136/200]: loss_d: 1.266, loss_g: 0.914
# [137/200]: loss_d: 1.275, loss_g: 0.894
# [138/200]: loss_d: 1.270, loss_g: 0.906
# [139/200]: loss_d: 1.277, loss_g: 0.890
# [140/200]: loss_d: 1.271, loss_g: 0.917
# [141/200]: loss_d: 1.268, loss_g: 0.907
# [142/200]: loss_d: 1.272, loss_g: 0.906
# [143/200]: loss_d: 1.277, loss_g: 0.893
# [144/200]: loss_d: 1.268, loss_g: 0.901
# [145/200]: loss_d: 1.277, loss_g: 0.897
# [146/200]: loss_d: 1.267, loss_g: 0.913
# [147/200]: loss_d: 1.269, loss_g: 0.900
# [148/200]: loss_d: 1.274, loss_g: 0.898
# [149/200]: loss_d: 1.275, loss_g: 0.892
# [150/200]: loss_d: 1.274, loss_g: 0.902
# [151/200]: loss_d: 1.269, loss_g: 0.907
# [152/200]: loss_d: 1.278, loss_g: 0.898
# [153/200]: loss_d: 1.282, loss_g: 0.884
# [154/200]: loss_d: 1.279, loss_g: 0.902
# [155/200]: loss_d: 1.278, loss_g: 0.901
# [156/200]: loss_d: 1.276, loss_g: 0.893
# [157/200]: loss_d: 1.276, loss_g: 0.897
# [158/200]: loss_d: 1.277, loss_g: 0.897
# [159/200]: loss_d: 1.280, loss_g: 0.896
# [160/200]: loss_d: 1.277, loss_g: 0.891
# [161/200]: loss_d: 1.276, loss_g: 0.903
# [162/200]: loss_d: 1.273, loss_g: 0.899
# [163/200]: loss_d: 1.278, loss_g: 0.894
# [164/200]: loss_d: 1.276, loss_g: 0.886
# [165/200]: loss_d: 1.279, loss_g: 0.891
# [166/200]: loss_d: 1.285, loss_g: 0.876
# [167/200]: loss_d: 1.278, loss_g: 0.902
# [168/200]: loss_d: 1.272, loss_g: 0.901
# [169/200]: loss_d: 1.271, loss_g: 0.898
# [170/200]: loss_d: 1.278, loss_g: 0.880
# [171/200]: loss_d: 1.279, loss_g: 0.892
# [172/200]: loss_d: 1.278, loss_g: 0.886
# [173/200]: loss_d: 1.282, loss_g: 0.882
# [174/200]: loss_d: 1.280, loss_g: 0.891
# [175/200]: loss_d: 1.280, loss_g: 0.890
# [176/200]: loss_d: 1.279, loss_g: 0.888
# [177/200]: loss_d: 1.278, loss_g: 0.889
# [178/200]: loss_d: 1.282, loss_g: 0.879
# [179/200]: loss_d: 1.287, loss_g: 0.881
# [180/200]: loss_d: 1.285, loss_g: 0.878
# [181/200]: loss_d: 1.286, loss_g: 0.877
# [182/200]: loss_d: 1.283, loss_g: 0.885
# [183/200]: loss_d: 1.280, loss_g: 0.885
# [184/200]: loss_d: 1.286, loss_g: 0.876
# [185/200]: loss_d: 1.287, loss_g: 0.876
# [186/200]: loss_d: 1.284, loss_g: 0.884
# [187/200]: loss_d: 1.283, loss_g: 0.877
# [188/200]: loss_d: 1.289, loss_g: 0.876
# [189/200]: loss_d: 1.284, loss_g: 0.882
# [190/200]: loss_d: 1.287, loss_g: 0.879
# [191/200]: loss_d: 1.283, loss_g: 0.877
# [192/200]: loss_d: 1.288, loss_g: 0.881
# [193/200]: loss_d: 1.284, loss_g: 0.878
# [194/200]: loss_d: 1.285, loss_g: 0.883
# [195/200]: loss_d: 1.292, loss_g: 0.869
# [196/200]: loss_d: 1.289, loss_g: 0.870
# [197/200]: loss_d: 1.293, loss_g: 0.862
# [198/200]: loss_d: 1.289, loss_g: 0.878
# [199/200]: loss_d: 1.282, loss_g: 0.883
# [200/200]: loss_d: 1.288, loss_g: 0.876
