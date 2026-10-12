import math
import torch
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
torch.set_default_dtype(torch.float64)

def sampling_points_random(N_inner=10000, N_boundary=400, x_min: float = -1.0, x_max: float = 1.0, y_min: float = -1.0, y_max: float = 1.0):
    """
    生成矩形域 [x_min, x_max] × [y_min, y_max] 的内部点和边界点（内部与边界严格分离）
    Args:
        N_inner (int): 内部区域采样点数
        N_boundary (int): 边界总采样点数；余数依次分配到四条边
        x_min, x_max, y_min, y_max (float): 采样区域边界
    Returns:
        coords_in (Tensor): 内部点坐标，形状 (N_inner, 2)
        coords_bc (Tensor): 边界点坐标，形状 (N_boundary, 2)
        coords_all (Tensor): 合并后的所有点，形状 (N_inner + N_boundary, 2)
    """
    # 输入校验
    assert N_inner >= 0 and N_boundary >= 0, "Sample counts must be nonnegative"
    assert x_max > x_min and y_max > y_min, "Invalid bounds: require x_max > x_min and y_max > y_min"

    # 内部点：严格在开区间内，排除边界
    span_x = x_max - x_min
    span_y = y_max - y_min
    eps = 1e-8 * min(span_x, span_y)  # 相对极小偏移量，避免数值问题
    if 2 * eps >= min(span_x, span_y):
        eps = 0.0  # 极小区域时不再收缩

    x_in = torch.empty(N_inner, 1, dtype=torch.float64).uniform_(x_min + eps, x_max - eps)
    y_in = torch.empty(N_inner, 1, dtype=torch.float64).uniform_(y_min + eps, y_max - eps)
    coords_in = torch.cat([x_in, y_in], dim=1)

    # 边界点尽可能均匀地分配到四条边；例如 250 -> 63+63+62+62。
    n_per_edge, remainder = divmod(N_boundary, 4)
    n_left, n_right, n_top, n_bottom = [
        n_per_edge + int(edge < remainder) for edge in range(4)
    ]

    # 左边 (x = x_min, y ∈ [y_min, y_max])
    left_y = torch.empty(n_left, 1, dtype=torch.float64).uniform_(y_min, y_max)
    left_coords = torch.cat([torch.full_like(left_y, x_min), left_y], dim=1)

    # 右边 (x = x_max, y ∈ [y_min, y_max])
    right_y = torch.empty(n_right, 1, dtype=torch.float64).uniform_(y_min, y_max)
    right_coords = torch.cat([torch.full_like(right_y, x_max), right_y], dim=1)

    # 上边 (y = y_max, x ∈ [x_min, x_max])
    top_x = torch.empty(n_top, 1, dtype=torch.float64).uniform_(x_min, x_max)
    top_coords = torch.cat([top_x, torch.full_like(top_x, y_max)], dim=1)

    # 下边 (y = y_min, x ∈ [x_min, x_max])
    bottom_x = torch.empty(n_bottom, 1, dtype=torch.float64).uniform_(x_min, x_max)
    bottom_coords = torch.cat([bottom_x, torch.full_like(bottom_x, y_min)], dim=1)

    coords_bc = torch.cat([left_coords, right_coords, top_coords, bottom_coords], dim=0)

    # 合并所有点（内部+边界）
    coords_all = torch.cat([coords_in, coords_bc], dim=0)

    coords_in = coords_in.to(device).requires_grad_(True)
    coords_bc = coords_bc.to(device).requires_grad_(True)
    coords_all = coords_all.to(device).requires_grad_(True)

    return coords_in, coords_bc, coords_all

def sampling_points_test(N_inner=10000):
    # 内部点：严格在(-1,1)区间内，排除边界
    eps = 1e-6  # 极小偏移量，避免数值问题
    coords_in = torch.empty(N_inner, 2, dtype=torch.float64).uniform_(-1 , 1 )
    coords_in = coords_in.to(device).requires_grad_(True).to(torch.float64)
    return coords_in

def sampling_points_linspace_test(N_inner=10000):
    # 内部点：严格在(-1,1)区间内，排除边界
    grid_size = int(math.sqrt(N_inner))
    linspace = torch.linspace(-1 , 1 , grid_size)
    x, y = torch.meshgrid(linspace, linspace, indexing='ij')
    coords = torch.stack([x.flatten(), y.flatten()], dim=1)
    coords = coords.to(device).requires_grad_(True).to(torch.float64)
    return coords

def sampling_points_linspace(N_inner=10000, N_boundary=400, x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0):
    """
    生成矩形域 [x_min, x_max] × [y_min, y_max] 内部点和边界点的正交网格（内部与边界严格分离）。
    Args:
        N_inner (int): 内部区域采样点数（将调整为最接近的完全平方数：grid_size^2）
        N_boundary (int): 边界总采样点数（需为4的倍数，四条边等分）
        x_min (float): 采样区域 x 下界
        x_max (float): 采样区域 x 上界
        y_min (float): 采样区域 y 下界
        y_max (float): 采样区域 y 上界
    Returns:
        coords_in (Tensor): 内部点坐标，形状 (grid_size^2, 2)
        coords_bc (Tensor): 边界点坐标，形状 (N_boundary, 2)
        coords_all (Tensor): 合并后的所有点，形状 (grid_size^2 + N_boundary, 2)
    """
    # 校验输入
    assert N_boundary % 4 == 0, "N_boundary must be divisible by 4"
    assert x_max > x_min and y_max > y_min, "Invalid bounds: require x_max > x_min and y_max > y_min"

    # 调整 N_inner 为最接近的完全平方数（向下取整）
    grid_size = int(math.sqrt(max(1, N_inner)))
    adjusted_N_inner = grid_size * grid_size  # 仅用于说明返回规模

    # 内部点：在 (x_min, x_max) × (y_min, y_max) 内生成正交网格，排除边界
    span_x = x_max - x_min
    span_y = y_max - y_min
    eps = 1e-3 * min(span_x, span_y)  # 相对极小偏移量，避免数值问题
    if 2 * eps >= min(span_x, span_y):
        eps = 0.0  # 极小区域时不再收缩

    lx = torch.linspace(x_min + eps, x_max - eps, grid_size, dtype=torch.float64)
    ly = torch.linspace(y_min + eps, y_max - eps, grid_size, dtype=torch.float64)
    x, y = torch.meshgrid(lx, ly, indexing='ij')
    coords_in = torch.stack([x.flatten(), y.flatten()], dim=1)

    # 边界点：四个边均匀采样（每条边 N_boundary//4 个点），端点按各边分别包含，可能在拼接后出现角点重复
    n_per_edge = N_boundary // 4

    # 左边 (x = x_min, y ∈ [y_min, y_max])
    left_y = torch.linspace(y_min, y_max, n_per_edge, dtype=torch.float64).unsqueeze(1)
    left_coords = torch.cat([torch.full_like(left_y, x_min), left_y], dim=1)

    # 右边 (x = x_max, y ∈ [y_min, y_max])
    right_y = torch.linspace(y_min, y_max, n_per_edge, dtype=torch.float64).unsqueeze(1)
    right_coords = torch.cat([torch.full_like(right_y, x_max), right_y], dim=1)

    # 上边 (y = y_max, x ∈ [x_min, x_max])
    top_x = torch.linspace(x_min, x_max, n_per_edge, dtype=torch.float64).unsqueeze(1)
    top_coords = torch.cat([top_x, torch.full_like(top_x, y_max)], dim=1)

    # 下边 (y = y_min, x ∈ [x_min, x_max])
    bottom_x = torch.linspace(x_min, x_max, n_per_edge, dtype=torch.float64).unsqueeze(1)
    bottom_coords = torch.cat([bottom_x, torch.full_like(bottom_x, y_min)], dim=1)

    # 合并边界点与全部点
    coords_bc = torch.cat([left_coords, right_coords, top_coords, bottom_coords], dim=0)
    coords_all = torch.cat([coords_in, coords_bc], dim=0)

    # 移动到设备并设置 requires_grad（使用全局 device）
    coords_in = coords_in.to(device).requires_grad_(True).to(torch.float64)
    coords_bc = coords_bc.to(device).requires_grad_(True).to(torch.float64)
    coords_all = coords_all.to(device).requires_grad_(True).to(torch.float64)

    return coords_in, coords_bc, coords_all

def sample_circle_interface(N_interface=200, r=0.5, center=(0.0, 0.0)):
    """
    仅生成圆形界面上的均匀分布采样点

    Args:
        N_interface (int): 圆形界面上的采样点数
        r (float): 圆的半径,默认为0.5
        center (tuple): 圆心坐标，默认为原点(0.0, 0.0)

    Returns:
        coords_interface (Tensor): 界面点坐标，形状 (N_interface, 2)
    """
    # 生成均匀分布的角度
    theta = torch.linspace(0, 2 * math.pi, N_interface + 1)[:-1]  # 去掉重复的终点

    # 计算圆上的坐标点
    x_interface = center[0] + r * torch.cos(theta)
    y_interface = center[1] + r * torch.sin(theta)

    # 组合为坐标张量
    coords_interface = torch.stack([x_interface, y_interface], dim=1)

    # 移动到设备并设置requires_grad
    coords_interface = coords_interface.to(device).requires_grad_(True).to(torch.float64)

    return coords_interface

def sample_flower_interface(N_interface=250):
    import math

    # 生成均匀分布的角度
    theta = torch.linspace(0, 2 * math.pi, N_interface + 1)[:-1]  # 去掉重复的终点

    r_flower = 0.6-0.2*torch.cos(7*theta)

    # 转换为笛卡尔坐标
    x_interface = r_flower * torch.cos(theta)
    y_interface = r_flower * torch.sin(theta)

    # 组合为坐标张量
    coords_interface = torch.stack([x_interface, y_interface], dim=1)

    # 移动到设备并设置requires_grad
    coords_interface = coords_interface.to(device).requires_grad_(True).to(torch.float64)

    return coords_interface

def sample_sharpcorner_interface(N_interface=250):
    import math

    # 生成均匀分布的角度
    theta = torch.linspace(0, 2 * math.pi, N_interface + 1)[:-1]  # 去掉重复的终点

    r_flower = 0.4+0.4*torch.sin(theta)

    # 转换为笛卡尔坐标
    x_interface = r_flower * torch.cos(theta)
    y_interface = r_flower * torch.sin(theta)

    # 组合为坐标张量
    coords_interface = torch.stack([x_interface, y_interface], dim=1)

    # 移动到设备并设置requires_grad
    coords_interface = coords_interface.to(device).requires_grad_(True).to(torch.float64)

    return coords_interface

def sample_ellipse_interface(N_interface=250):
    """
    生成椭圆界面上的均匀分布采样点
    椭圆方程：x²/0.7² + y²/0.5² = 1

    Args:
        N_interface (int): 界面上的采样点数

    Returns:
        coords_interface (Tensor): 界面点坐标，形状 (N_interface, 2)
    """
    import math

    # 椭圆参数
    a = 0.7  # x轴半长轴
    b = 0.5  # y轴半长轴

    # 生成均匀分布的角度
    theta = torch.linspace(0, 2 * math.pi, N_interface + 1)[:-1]  # 去掉重复的终点

    # 计算椭圆边界上的点坐标
    x_interface = a * torch.cos(theta)
    y_interface = b * torch.sin(theta)

    # 组合为坐标张量
    coords_interface = torch.stack([x_interface, y_interface], dim=1)

    # 移动到设备并设置requires_grad
    coords_interface = coords_interface.to(device).requires_grad_(True).to(torch.float64)

    return coords_interface


def databuild(x_in, x_bc, x_interface, normal, sign, device=device, K:int=3):
    """构建高斯混合界面算例的数据

    方程: -∇·(β∇u) = f,  Ω=[-1,1]^2 \ Γ
    界面条件: [u]=w, [β∂_n u]=v,  边界: u=g

    β(x,y)=1+(1/7)cos(π(x-y));
    u^-(x,y)=sin(Kπx)sin(Kπy) (phi<=0),  u^+(x,y)=cos(Kπx)cos(Kπy) (phi>0)
    """
    x_dict = {}
    data_dict = {}

    # variable coefficient beta and its gradient
    x = x_in[:, 0:1]
    y = x_in[:, 1:2]
    beta = 1.0 + (1.0/7.0)*torch.cos(math.pi*(x - y))
    dbdx = -(math.pi/7.0)*torch.sin(math.pi*(x - y))
    dbdy =  (math.pi/7.0)*torch.sin(math.pi*(x - y))
    grad_beta = torch.cat([dbdx, dbdy], dim=1)

    # exact solution pieces and derivatives
    Kpi = K*math.pi
    # inside (phi<=0): sin-sin
    u_in = torch.sin(Kpi*x)*torch.sin(Kpi*y)
    ux_in = Kpi*torch.cos(Kpi*x)*torch.sin(Kpi*y)
    uy_in = Kpi*torch.sin(Kpi*x)*torch.cos(Kpi*y)
    lap_in = -2*(Kpi**2)*torch.sin(Kpi*x)*torch.sin(Kpi*y)

    # outside (phi>0): cos-cos
    u_out = torch.cos(Kpi*x)*torch.cos(Kpi*y)
    ux_out = -Kpi*torch.sin(Kpi*x)*torch.cos(Kpi*y)
    uy_out = -Kpi*torch.cos(Kpi*x)*torch.sin(Kpi*y)
    lap_out = -2*(Kpi**2)*torch.cos(Kpi*x)*torch.cos(Kpi*y)

    # choose piece by sign (1:inside, 0:outside) – but here only to build RHS
    inside_mask = sign > 0.5
    grad_u_x = torch.where(inside_mask, ux_in, ux_out)
    grad_u_y = torch.where(inside_mask, uy_in, uy_out)
    lap_u    = torch.where(inside_mask, lap_in, lap_out)

    # f = -beta Δu - grad(beta)·grad(u)
    f = -beta*lap_u - (dbdx*grad_u_x + dbdy*grad_u_y)

    # Boundary Dirichlet g = exact piecewise u
    xb = x_bc[:, 0:1]
    yb = x_bc[:, 1:2]
    beta_b = 1.0 + (1.0/7.0)*torch.cos(math.pi*(xb - yb))
    # needed on interface too
    # piecewise u on boundary
    # we require sign on boundary as well; recompute via provided convention using level-set outside
    # For generality, caller should provide sign only for inner points; here we apply a heuristic using the same K
    # but g_D uses exact u^+ if outside and u^- if inside; we'll estimate with phi at x_bc in trainer.
    # To avoid dependency here, assume outside by default then overwrite in trainer; hence compute both and select later.
    g_in = torch.sin(Kpi*xb)*torch.sin(Kpi*yb)
    g_out = torch.cos(Kpi*xb)*torch.cos(Kpi*yb)
    # g_D will be finalized in trainer using actual phi on boundary; for now choose outside by default then trainer may regenerate.
    g_D = g_out

    # Interface data: jumps
    xi = x_interface[:, 0:1]
    yi = x_interface[:, 1:2]
    beta_i = 1.0 + (1.0/7.0)*torch.cos(math.pi*(xi - yi))
    # values on both sides
    u_minus = torch.sin(Kpi*xi)*torch.sin(Kpi*yi)   # phi<=0
    u_plus  = torch.cos(Kpi*xi)*torch.cos(Kpi*yi)   # phi>0
    w = u_plus - u_minus

    # normal flux jump v = beta(∂n u^+ - ∂n u^-)
    ux_minus = Kpi*torch.cos(Kpi*xi)*torch.sin(Kpi*yi)
    uy_minus = Kpi*torch.sin(Kpi*xi)*torch.cos(Kpi*yi)
    ux_plus  = -Kpi*torch.sin(Kpi*xi)*torch.cos(Kpi*yi)
    uy_plus  = -Kpi*torch.cos(Kpi*xi)*torch.sin(Kpi*yi)
    grad_minus_n = (ux_minus*normal[:,0:1] + uy_minus*normal[:,1:2])
    grad_plus_n  = (ux_plus *normal[:,0:1] + uy_plus *normal[:,1:2])
    v = beta_i*(grad_plus_n - grad_minus_n)

    # pack dicts
    x_dict["inner"] = x_in
    x_dict["boundary"] = x_bc
    x_dict["interface"] = x_interface

    data_dict["inner"] = {
        "f": f.to(device),
        "sign": sign,
        "beta": beta.to(device),
        "grad_beta": grad_beta.to(device)
    }
    data_dict["boundary"] = {"g_D": g_D.to(device)}
    data_dict["interface"] = {
        "w": w.to(device),
        "v": v.to(device),
        "normal_vector": normal,
        "ibeta1": beta_i.to(device),
        "ibeta2": beta_i.to(device)
    }

    return x_dict, data_dict

def Greenfunction(x_test,sign,device=device):
    """
    保留接口以兼容旧代码；本算例不包含奇异解，返回全零。
    """
    return torch.zeros_like(x_test[:,0:1], dtype=torch.float64, device=device)
