import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        """初始化KANLinear层

        KANLinear是实现Kolmogorov-Arnold网络(KAN)的核心组件，它结合了传统线性层和
        B样条表示，提供了强大的非线性建模能力。

        参数:
            in_features (int): 输入特征维度
            out_features (int): 输出特征维度
            grid_size (int): B样条表示的网格点数量，越大表示能力越强
            spline_order (int): B样条的阶数，越高生成的样条越平滑
            scale_noise (float): 初始化样条权重时的随机噪声缩放系数
            scale_base (float): 基础权重的缩放系数
            scale_spline (float): 样条权重的缩放系数
            enable_standalone_scale_spline (bool): 是否使用独立的样条缩放参数
            base_activation (torch.nn.Module): 基础路径使用的激活函数
            grid_eps (float): 控制网格更新时自适应与均匀网格的平衡系数
            grid_range (list): B样条覆盖的输入范围，格式为[最小值, 最大值]
        """
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 计算网格步长
        h = (grid_range[1] - grid_range[0]) / grid_size
        # 创建B样条的网格节点，包含扩展边界点以处理边界条件
        # 形状: (in_features, grid_size + 2*spline_order + 1)
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid) # 注册为非参数缓冲区

        # 基础线性层权重，形状: (out_features, in_features)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # B样条权重系数，形状: (out_features, in_features, grid_size + spline_order)
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        # 可选的独立样条缩放因子，形状: (out_features, in_features)
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        # 存储配置参数
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation() # 实例化激活函数
        self.grid_eps = grid_eps

        # 初始化所有参数
        self.reset_parameters()

    def reset_parameters(self):
        """初始化KANLinear层的参数
        该方法负责初始化模型的三个主要权重参数：
        1. base_weight: 基础线性层权重，使用Kaiming均匀分布初始化
        2. spline_weight: B样条权重系数，通过随机噪声和curve2coeff函数初始化
        3. spline_scaler: 样条缩放因子(如果启用)，使用Kaiming均匀分布初始化

        初始化步骤：
        - 使用kaiming_uniform_初始化基础线性层权重，缩放因子为scale_base
        - 生成随机噪声张量，形状为(grid_size+1, in_features, out_features)
        - 使用curve2coeff将噪声转换为样条系数
        - 如果启用独立样条缩放，初始化样条缩放因子
        """
        # 使用Kaiming均匀分布初始化基础线性层权重
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # 创建小随机噪声作为初始化点
            # 形状: (grid_size+1, in_features, out_features)
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2 # 中心化在0周围
                )
                * self.scale_noise # 缩放噪声振幅
                / self.grid_size # 根据网格大小归一化
            )
            # 使用curve2coeff将噪声转换为B样条系数，并应用scale_spline缩放
            # grid.T[spline_order:-spline_order]提取网格的主体部分作为样条定义点
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            # 如果启用独立样条缩放，初始化缩放因子
            if self.enable_standalone_scale_spline:
                # 注释掉的代码是使用常数初始化的替代方法
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """计算输入张量的B样条基函数

        该方法实现了Cox-de Boor递归公式来计算B样条基函数。B样条是KAN网络实现非线性变换的核心，
        它提供了局部支持的平滑基函数集，可以组合表示复杂的非线性映射。

        实现步骤:
        1. 首先初始化0阶B样条(分段常数函数)，判断输入点落在哪个网格区间
        2. 然后通过递归公式依次计算更高阶的B样条基函数，直到达到指定的spline_order
        3. 每一阶的递归计算都基于两个加权的低阶基函数，权重由输入值与网格点的相对位置决定

        Cox-de Boor递归公式:
        B_{i,0}(x) = 1 如果 t_i ≤ x < t_{i+1}，否则为0
        B_{i,k}(x) = (x-t_i)/(t_{i+k}-t_i) * B_{i,k-1}(x) + (t_{i+k+1}-x)/(t_{i+k+1}-t_{i+1}) * B_{i+1,k-1}(x)

        参数:
            x (torch.Tensor): 输入张量，形状为(batch_size, in_features)

        返回:
            torch.Tensor: B样条基函数张量，形状为(batch_size, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1) # 扩展维度用于广播计算，形状变为(batch_size, in_features, 1)

        # 初始化0阶B样条基函数：判断每个点落在哪个网格区间
        # 形状: (batch_size, in_features, grid_size + 2*spline_order)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # 通过Cox-de Boor递归公式计算更高阶B样条
        for k in range(1, self.spline_order + 1):
            # 每一步递归计算都融合了左右两侧的低阶基函数
            # 左侧项：(x-t_i)/(t_{i+k}-t_i) * B_{i,k-1}(x)
            # 右侧项：(t_{i+k+1}-x)/(t_{i+k+1}-t_{i+1}) * B_{i+1,k-1}(x)
            bases = (
                (x - grid[:, : -(k + 1)]) # 分子：x-t_i
                / (grid[:, k:-1] - grid[:, : -(k + 1)]) # 分母：t_{i+k}-t_i
                * bases[:, :, :-1] # 乘以B_{i,k-1}(x)
            ) + (
                (grid[:, k + 1 :] - x) # 分子：t_{i+k+1}-x
                / (grid[:, k + 1 :] - grid[:, 1:(-k)]) # 分母：t_{i+k+1}-t_{i+1}
                * bases[:, :, 1:] # 乘以B_{i+1,k-1}(x)
            )

        # 验证输出形状
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous() # 确保内存连续以优化后续计算

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=15,
        spline_order=6,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

# 新增：基于 KAN 的“乘积-求和”结构
class KANProductSum(torch.nn.Module):
    """
    y(x) = sum_{j=1}^p C_{t,j} * prod_{i=1}^d phi_{i,j}(x_i),  t=1..out_features
    - 使用 KAN 作为每个维度的一维子网络 phi_i: R->R^p
    """
    def __init__(
        self,
        d: int,
        p: int,
        hidden: int = 5,
        depth: int = 2,
        out_features: int = 2,
        # KAN 超参
        grid_size: int = 15,
        spline_order: int = 6,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.Tanh,
        grid_eps: float = 0.02,
        grid_range=[-1, 1],
        l2_normalize: bool = False,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.d = d
        self.p = p
        self.out_features = out_features
        self.l2_normalize = l2_normalize
        self.norm_eps = eps

        # 每个维度一套 1D KAN，输出维度为 p
        layers_hidden = [1] + [hidden] * depth + [p]
        self.phi = torch.nn.ModuleList([
            KAN(
                layers_hidden=layers_hidden,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            for _ in range(d)
        ])

        # 线性组合系数矩阵 C: (out_features, p)
        self.C = torch.nn.Parameter(torch.zeros(out_features, p))
        torch.nn.init.normal_(self.C, std=0.1)

    def forward(self, x: torch.Tensor, derivative_order=0):
        if derivative_order:
            from .tenkan_derivatives import product_sum_jet
            return product_sum_jet(self, x, derivative_order)
        # 兼容 (d,) 与 (N,d)
        single_sample = False
        if x.dim() == 1:            # (d,) -> (1,d)
            x = x.unsqueeze(0)
            single_sample = True
        elif x.dim() > 2:           # 其他情况展平成 (N,d)
            x = x.reshape(-1, self.d)

        feats = []
        for i in range(self.d):
            xi = x[:, i:i+1]            # (N,1)
            feats.append(self.phi[i](xi))  # (N,p)
        Phi = torch.stack(feats, dim=1)    # (N,d,p)

        # 可选：按批维进行 L2 归一化，提升数值稳定性（论文中的 φ̂）
        if self.l2_normalize:
            # 对每个 (i,j) 计算 RMS: sqrt(mean_n φ^2) -> (1,d,p)
            rms = torch.sqrt(torch.mean(Phi.pow(2), dim=0, keepdim=True) + self.norm_eps)
            Phi = Phi / rms

        prod_over_dim = Phi.prod(dim=1)        # (N,p)
        y = prod_over_dim @ self.C.t()         # (N,out_features)

        # 仅去掉 batch 维，避免变成 0 维标量
        return y.squeeze(0) if single_sample else y
