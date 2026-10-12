# 在测试代码后添加以下内容
# 计算beta系数的真实值和误差
import torch
import math

def compute_errors(x_test, u_pred,u_test, phi,sign):
    """计算测试集的l2误差"""


    outside_mask = phi > 0.0  # 界面外的点 (phi>0表示圆外)

    # 计算总体L2误差

    l2_error = torch.sqrt(torch.mean((u_pred - u_test)**2))

    # 计算相对L2误差
    rel_l2_error = torch.sqrt(torch.sum((u_pred - u_test)**2) / torch.sum(u_test**2))

    # 计算L无穷误差（最大绝对误差）
    linf_error = torch.max(torch.abs(u_pred - u_test))

    # 计算相对L无穷误差
    rel_linf_error = torch.max(torch.abs(u_pred - u_test)) / torch.max(torch.abs(u_test))

    # 分别计算界面内部和外部的误差
    l2_error_inside = torch.sqrt(torch.mean((u_pred[~outside_mask] - u_test[~outside_mask])**2))
    l2_error_outside = torch.sqrt(torch.mean((u_pred[outside_mask] - u_test[outside_mask])**2))

    # 计算界面附近区域的误差
    interface_region = (phi.abs() < 0.025).squeeze()  # 选择水平集函数接近零的点
    if interface_region.sum() > 0:  # 确保有界面点
        l2_error_interface = torch.sqrt(torch.mean((u_pred[interface_region] - u_test[interface_region])**2))
    else:
        l2_error_interface = torch.tensor(float('nan'))

    # 计算点点误差用于可视化
    point_errors = torch.abs(u_pred - u_test)

    return  linf_error, rel_linf_error, l2_error, rel_l2_error, l2_error_inside, l2_error_outside, l2_error_interface, point_errors

def compute_errors_anly(x_test, u_pred, phi=None, K:int=3):
    """
    计算测试集上的预测解与解析解之间的误差
    Args:
        x_test: 测试点坐标
        u_pred: 预测解
        u_test: 可选的测试数据（本函数不使用，保持参数兼容性）
        phi: 水平集函数值，用于区分内外区域

    Returns:
        各种误差指标
    """
    # 解析解：
    # Ω1 (phi<=0): sin(Kπx)sin(Kπy)
    # Ω2 (phi>0):  cos(Kπx)cos(Kπy)
    x = x_test[:, 0]
    y = x_test[:, 1]

    assert phi.dim() == 2, "phi should be a 1D tensor"
    assert u_pred.dim() == 2, "u_pred should be a 2D tensor with shape [N,1]"

    # 区分内外区域
    outside_mask = (phi > 0.0).squeeze()  # phi>0 表示Ω2

    # 根据区域生成解析解
    u_analytic = torch.zeros_like(u_pred)
    # 确保下标操作正确
    if u_analytic.dim() > 1:  # 如果u_analytic是二维的
        # 应用内部区域的解析解
        if (~outside_mask).any():
            inside_indices = torch.where(~outside_mask)[0]
            u_analytic[inside_indices, :] = (
                torch.sin(K*math.pi*x[inside_indices]) * torch.sin(K*math.pi*y[inside_indices])
            ).unsqueeze(1)

        # 应用外部区域的解析解
        if outside_mask.any():
            outside_indices = torch.where(outside_mask)[0]
            u_analytic[outside_indices, :] = (
                torch.cos(K*math.pi*x[outside_indices]) * torch.cos(K*math.pi*y[outside_indices])
            ).unsqueeze(1)
    else:  # 如果u_analytic是一维的
        # 应用内部区域的解析解
        if (~outside_mask).any():
            u_analytic[~outside_mask] = torch.sin(K*math.pi*x[~outside_mask]) * torch.sin(K*math.pi*y[~outside_mask])

        # 应用外部区域的解析解
        if outside_mask.any():
            u_analytic[outside_mask] = torch.cos(K*math.pi*x[outside_mask]) * torch.cos(K*math.pi*y[outside_mask])

    # 计算总体L2误差
    l2_error = torch.sqrt(torch.mean((u_pred - u_analytic)**2))

    # 计算相对L2误差
    rel_l2_error = torch.sqrt(torch.sum((u_pred - u_analytic)**2) / torch.sum((u_analytic)**2))

    # 计算L无穷误差（最大绝对误差）
    linf_error = torch.max(torch.abs(u_pred - u_analytic))

    # 计算相对L无穷误差
    rel_linf_error = torch.max(torch.abs(u_pred - u_analytic)) / torch.max(torch.abs(u_analytic))

    # 分别计算界面内部和外部的误差
    l2_error_inside = torch.sqrt(torch.mean((u_pred[~outside_mask] - u_analytic[~outside_mask])**2))
    l2_error_outside = torch.sqrt(torch.mean((u_pred[outside_mask] - u_analytic[outside_mask])**2))

    # 计算界面附近区域的误差
    interface_region = (phi.abs() < 0.02).squeeze()  # 选择水平集函数接近零的点
    if interface_region.sum() > 0:  # 确保有界面点
        l2_error_interface = torch.sqrt(torch.mean((u_pred[interface_region] - u_analytic[interface_region])**2))
    else:
        l2_error_interface = torch.tensor(float('nan'))

    # 计算点点误差用于可视化
    point_errors = torch.abs(u_pred - u_analytic)

    return linf_error, rel_linf_error, l2_error, rel_l2_error, l2_error_inside, l2_error_outside, l2_error_interface, point_errors


def compute_broken_h1_anly(model_fn, x_test, phi, K: int = 3,
                           batch_size: int = 8192,
                           interface_tol: float = 1e-8):
    r"""Compute discrete relative broken-:math:`H^1` errors.

    On the uniformly weighted test points in the two open subdomains this uses

    .. math::
       \frac{\left(\sum_i |e_i|^2 + |\nabla e_i|^2\right)^{1/2}}
            {\left(\sum_i |u_i|^2 + |\nabla u_i|^2\right)^{1/2}}.

    Points satisfying ``abs(phi) <= interface_tol`` are excluded because a
    broken Sobolev norm is defined separately on the open subdomains and the
    discontinuous exact solution has no single classical interface gradient.
    Autograd is applied to ``U + I_{phi<=0} V`` while the region indicator is
    held fixed, so no distributional derivative of the indicator is included.
    Batching keeps the additional derivative evaluation practical on a GPU.
    """
    if x_test.ndim != 2 or x_test.shape[1] != 2:
        raise ValueError("x_test must have shape [N, 2]")
    if phi.shape[0] != x_test.shape[0]:
        raise ValueError("phi and x_test must contain the same number of points")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    value_error_sq = 0.0
    gradient_error_sq = 0.0
    exact_value_sq = 0.0
    exact_gradient_sq = 0.0
    sample_count = 0
    omega = K * math.pi

    for start in range(0, x_test.shape[0], batch_size):
        stop = min(start + batch_size, x_test.shape[0])
        x = x_test[start:stop].detach().clone().requires_grad_(True)
        phi_batch = phi[start:stop].detach().reshape(-1)
        valid = phi_batch.abs() > interface_tol
        if not valid.any():
            continue

        with torch.enable_grad():
            output = model_fn(x)
            indicator_minus = (phi_batch <= 0.0).to(output.dtype).unsqueeze(1)
            u_pred = output[:, 0:1] + indicator_minus * output[:, 1:2]
            grad_pred = torch.autograd.grad(
                u_pred, x, grad_outputs=torch.ones_like(u_pred),
                create_graph=False, retain_graph=False
            )[0]

        xx = x[:, 0]
        yy = x[:, 1]
        in_minus = phi_batch <= 0.0
        u_minus = torch.sin(omega * xx) * torch.sin(omega * yy)
        u_plus = torch.cos(omega * xx) * torch.cos(omega * yy)
        ux_minus = omega * torch.cos(omega * xx) * torch.sin(omega * yy)
        uy_minus = omega * torch.sin(omega * xx) * torch.cos(omega * yy)
        ux_plus = -omega * torch.sin(omega * xx) * torch.cos(omega * yy)
        uy_plus = -omega * torch.cos(omega * xx) * torch.sin(omega * yy)
        u_exact = torch.where(in_minus, u_minus, u_plus).unsqueeze(1)
        grad_exact = torch.stack(
            [torch.where(in_minus, ux_minus, ux_plus),
             torch.where(in_minus, uy_minus, uy_plus)], dim=1
        )

        value_diff = u_pred.detach()[valid] - u_exact.detach()[valid]
        gradient_diff = grad_pred.detach()[valid] - grad_exact.detach()[valid]
        value_error_sq += float(torch.sum(value_diff.square()).cpu())
        gradient_error_sq += float(torch.sum(gradient_diff.square()).cpu())
        exact_value_sq += float(torch.sum(u_exact.detach()[valid].square()).cpu())
        exact_gradient_sq += float(torch.sum(grad_exact.detach()[valid].square()).cpu())
        sample_count += int(valid.sum().item())

    if sample_count == 0:
        raise ValueError("No test points remain after excluding the interface")

    h1_error_sq = value_error_sq + gradient_error_sq
    exact_h1_sq = exact_value_sq + exact_gradient_sq
    return {
        "broken_h1_rms": math.sqrt(h1_error_sq / sample_count),
        "relative_broken_h1": math.sqrt(h1_error_sq / exact_h1_sq),
        "broken_h1_seminorm_rms": math.sqrt(gradient_error_sq / sample_count),
        "relative_broken_h1_seminorm": math.sqrt(gradient_error_sq / exact_gradient_sq),
        "broken_h1_test_points": sample_count,
        "excluded_interface_points": int(x_test.shape[0]) - sample_count,
    }
