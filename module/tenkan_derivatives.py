"""Differentiable spatial jets for separable KANs (no parameter detaches).

The knot convention matches KANLinear, including its right-sided evaluation
at knots. Only Tanh base activations and unnormalised encoders are supported.
"""
import torch
import torch.nn.functional as F


def spline_jet(layer, x, order):
    grid = layer.grid
    z = x.unsqueeze(-1)
    b = ((z >= grid[:, :-1]) & (z < grid[:, 1:])).to(x.dtype)
    db = torch.zeros_like(b)
    ddb = torch.zeros_like(b)
    q = layer.spline_order
    for k in range(1, q + 1):
        left = grid[:, k:-1] - grid[:, :-(k + 1)]
        right = grid[:, k + 1:] - grid[:, 1:-k]
        if k == q and order == 2:
            ddb = k * (db[:, :, :-1] / left - db[:, :, 1:] / right)
        if k == q or (order == 2 and k == q - 1):
            db = k * (b[:, :, :-1] / left - b[:, :, 1:] / right)
        b = ((z - grid[:, :-(k + 1)]) / left * b[:, :, :-1]
             + (grid[:, k + 1:] - z) / right * b[:, :, 1:])
    return b, db, ddb


def encoder_jet(encoder, x, order):
    dx = torch.ones_like(x)
    ddx = torch.zeros_like(x)
    for layer in encoder.layers:
        if not isinstance(layer.base_activation, torch.nn.Tanh):
            raise ValueError("TenKAN spatial jets require Tanh base activations")
        b, db, ddb = spline_jet(layer, x, order)
        w = layer.scaled_spline_weight.reshape(layer.out_features, -1)
        a = torch.tanh(x)
        da = 1 - a.square()
        value = F.linear(a, layer.base_weight) + F.linear(b.flatten(1), w)
        first = (F.linear(da * dx, layer.base_weight)
                 + F.linear((db * dx.unsqueeze(-1)).flatten(1), w))
        if order == 2:
            second = (
                F.linear(-2 * a * da * dx.square() + da * ddx, layer.base_weight)
                + F.linear((ddb * dx.square().unsqueeze(-1)
                            + db * ddx.unsqueeze(-1)).flatten(1), w)
            )
            ddx = second
        x, dx = value, first
    return x, dx, ddx


def product_sum_jet(model, x, order):
    """Return (value, gradient, laplacian); gradient axes are (..., output, d)."""
    if order not in (1, 2):
        raise ValueError("derivative_order must be 1 or 2")
    if model.l2_normalize:
        raise ValueError("Spatial jets do not support batch-dependent normalization")
    single = x.ndim == 1
    x = x.reshape(-1, model.d)
    jets = [encoder_jet(net, x[:, i:i + 1], order)
            for i, net in enumerate(model.phi)]
    values = torch.stack([jet[0] for jet in jets], dim=1)
    y = values.prod(dim=1) @ model.C.t()
    grads, seconds = [], []
    for i in range(model.d):
        # Never divide by a factor: zero-valued factors are valid.
        others = torch.ones_like(jets[i][0])
        for j in range(model.d):
            if j != i:
                others = others * jets[j][0]
        grads.append((jets[i][1] * others) @ model.C.t())
        if order == 2:
            seconds.append((jets[i][2] * others) @ model.C.t())
    gradient = torch.stack(grads, dim=-1)
    laplacian = torch.stack(seconds, dim=-1).sum(-1) if order == 2 else None
    if single:
        return y.squeeze(0), gradient.squeeze(0), (laplacian.squeeze(0) if order == 2 else None)
    return y, gradient, laplacian
