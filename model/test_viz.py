import torch
from model import UNet
from torchviz import make_dot

x = torch.randn(4, 1, 80, 80)
net = UNet(n_channels=1)
y = net(x)
make_dot(y, params=dict(list(net.named_parameters()))).render("net_torchviz", format="png")