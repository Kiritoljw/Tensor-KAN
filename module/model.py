import torch
import torch.nn as nn
import numpy as np
torch.set_default_dtype(torch.float64)

class Shallow_Deep_ext(nn.Module):
    def __init__(self, in_dim , h_dim , out_dim, layers=4,addiction_features=None):
        super().__init__()
        self.in_dim=in_dim

        if addiction_features is None:
            self.with_af=False
            # self.ln1 = nn.Linear( in_dim , h_dim )
            layer_list=[self.in_dim]+[h_dim]*layers+[out_dim]
        else:
            self.with_af=True
            self.phi=lambda x: torch.stack([af(x) for af in addiction_features],dim=x.ndim-1)
            # self.ln1 = nn.Linear( in_dim+ len(addiction_features) , h_dim )
            layer_list=[self.in_dim+len(addiction_features)]+[h_dim]*layers+[out_dim]

        self.act =nn.Sigmoid()
        self.list=nn.ModuleList()
        for i in range(1,len(layer_list)):
            if i==len(layer_list)-1:
                self.list.append(nn.Linear( h_dim , out_dim , bias=False ))
            else:
                self.list.append(nn.Linear(layer_list[i-1],layer_list[i]))

    def forward(self, x):
        if x.shape[-1]==self.in_dim and self.with_af:
            x=torch.hstack([x,self.phi(x)])
        # otherwise x directly contains addition_features or none
        x = x0 = self.act(self.list[0](x))
        for i,layer in enumerate(self.list[1:-1]):
            x = self.act(layer(x))
        x = self.list[-1](x+x0)
        return x

class HRKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int, imin: float, imax: float, order = int, train_ab: bool = True):
        super().__init__()

        self.g, self.k, self.r = g, k, 2*g / ((k+1)*(imax-imin))
        self.order = order

        self.input_size, self.output_size = input_size, output_size

        phase_low = (imax-imin) * (np.arange(-k, g) / g) - (-imin) #定义核函数的下界
        phase_height = phase_low + (k+1) / g * (imax-imin) #定义核函数的上界

        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(input_size)])),
                                      requires_grad=train_ab)
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(input_size)])),
                                         requires_grad=train_ab)
        self.equal_size_conv = nn.Conv2d(1, output_size, (g+k, input_size))

    def forward(self, x):

        # Expand dimensions of x to match the shape of self.phase_low
        if x.ndim==2:
            x_expanded = x.unsqueeze(2).expand(-1, -1, self.phase_low.size(1))
            # Perform the subtraction with broadcasting
            x1 = torch.relu(x_expanded - self.phase_low)
            x2 = torch.relu(self.phase_height - x_expanded)

            x = (x1 * x2 * self.r)**(self.order)

            x = x * x
            x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
            x = self.equal_size_conv(x)
            x = x.reshape((len(x), self.output_size))
        else:
            x_expanded = x.unsqueeze(1).expand( -1, self.phase_low.size(1))
            # Perform the subtraction with broadcasting
            x1 = torch.relu(x_expanded - self.phase_low)
            x2 = torch.relu(self.phase_height - x_expanded)

            x = (x1 * x2 * self.r)**(self.order)

            x = x * x

            x = x.reshape(1, self.g + self.k, self.input_size)
            x = self.equal_size_conv(x)
            x = x.reshape((self.output_size))

        return x


class HRKAN(nn.Module):
    def __init__(self, width, grid, k, imin, imax, order):
        super().__init__()
        self.width = width
        self.grid = grid
        self.k = k
        self.rk_layers = []
        for i in range(len(width) - 1):
            self.rk_layers.append(HRKANLayer(width[i], grid, k, width[i+1], imin, imax, order))
        self.rk_layers = nn.ModuleList(self.rk_layers)

    def forward(self, x):
        for rk_layer in self.rk_layers:
            x = rk_layer(x)
        return x

class MLP1D(nn.Module):
    def __init__(self, in_features=1, hidden=64, out_features=16, depth=2, act=nn.Tanh):
        super().__init__()
        layers = []
        fin = in_features
        for _ in range(depth):
            layers += [nn.Linear(fin, hidden), act()]
            fin = hidden
        layers += [nn.Linear(fin, out_features)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (N, 1)
        return self.net(x)  # (N, p)

class TNNProductSum(nn.Module):
    """
    y(x) = sum_{j=1}^p c_j * prod_{i=1}^d phi_{i,j}(x_i)
    - d: 输入维度
    - p: 分离秩/基函数个数
    - 每个维度一个 1D MLP，输出 p 维特征
    """
    def __init__(self, d: int, p: int, hidden=64, depth=2, act=nn.Tanh, out_features: int = 2):
        super().__init__()
        self.d = d
        self.p = p
        self.phi = nn.ModuleList([MLP1D(1, hidden, p, depth, act) for _ in range(d)])
        # 可训练系数 c_j
        self.c = nn.Parameter(torch.zeros(out_features, p))
        nn.init.normal_(self.c, std=0.1)
        # L2 归一化的数值稳定项
        self.norm_eps = 1e-12

    def forward(self, x: torch.Tensor):
        # 支持 (d,) 或 (N,d)
        single_sample = False
        if x.dim() == 1:        # (d,) -> (1,d)
            x = x.unsqueeze(0)
            single_sample = True
        elif x.dim() > 2:       # 其他情况，压平成 (N,d)
            x = x.reshape(-1, self.d)

        feats = []
        for i in range(self.d):
            xi = x[:, i:i+1]                 # (N,1)
            feats.append(self.phi[i](xi))    # (N,p)
        Phi = torch.stack(feats, dim=1)      # (N,d,p)

        # # L2 归一化：对每个 (i,j) 沿批维 N 计算 ||·||_2
        # # 等价离散近似: ||φi,j||_L2 ≈ sqrt(mean_n φ(n)^2)
        # rms = torch.sqrt(torch.sum(Phi**2, dim=2) + self.norm_eps).unsqueeze(dim=-1)  # (1,d,p)
        # Phi_hat = Phi / rms                                                        # (N,d,p)
        prod_over_dim = Phi.prod(dim=1)      # (N,p)
        y = prod_over_dim @ self.c.t()       # (N,out_features)

        # 单样本时返回 (1,) 或标量，便于 vmap 叠加
        return y.squeeze() if single_sample else y