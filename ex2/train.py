import os
# 避免 Intel Fortran 运行库在 Ctrl-C 时直接终止进程（forrtl: error 200）
# 必须在导入 numpy/torch/scipy 等库之前设置
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import qmc
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import sys
import math
import argparse
import csv
import json
import platform
import signal
import socket
import time

# 添加父目录到Python路径，确保能找到module包
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data_gen import *
from loss_func import *
from module.model import *
from module.optimizer import *
from evaluate import *
from module.sampler import *
from module.kan import *
import datetime
from module.visualize import visualize_data


print("当前工作目录:", current_dir)
print("父目录路径:", parent_dir)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

DEFAULT_DTYPE = torch.float64
torch.set_default_dtype(DEFAULT_DTYPE)

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def sign_func(phi):
        """基于水平集函数判断点的区域指示函数"""
        return torch.where(phi <= 0.0,
                           torch.tensor(1.0, dtype=phi.dtype, device=phi.device),
                           torch.tensor(0.0, dtype=phi.dtype, device=phi.device))

# 统一的 Ctrl-C 处理：将控制台事件转换为 Python 的 KeyboardInterrupt
def _sigint_to_keyboardinterrupt(signum, frame):
    raise KeyboardInterrupt()

try:
    signal.signal(signal.SIGINT, _sigint_to_keyboardinterrupt)
except Exception:
    pass

class Trainer:
    def __init__(self, coords_in, coords_bc, coords_all,
                 model_type: str = 'TenKAN', optimizer_type: str = 'gdlm',
                 lm_iter: int = None, k: int = 4,
                 n_interface_desired: int = 160, tol: float = 5e-3,
                 dtype=None, adam_lr: float = 5e-3,
                 rank: int = 5, grid_size: int = 15,
                 spline_order: int = 6, kan_hidden: int = 5,
                 kan_depth: int = 2, seed: int = 1234,
                 interface_candidates: int = 200000,
                 test_points: int = 90000, eval_batch_size: int = 8192,
                 output_root: str = None, run_id: str = None,
                 save_plots: bool = True,
                 weight_bd: float = 1.0, weight_f: float = 1.0,
                 weight_jump: float = 1.0,
                 weight_normal_jump: float = 1.0,
                 tenkan_derivatives: str = "autograd"):
        # 初始化参数
        self.dtype = coords_in.dtype if dtype is None else dtype
        self.coords_in = coords_in.detach().to(device=device, dtype=self.dtype).requires_grad_(True)
        self.coords_bc = coords_bc.detach().to(device=device, dtype=self.dtype).requires_grad_(True)
        self.coords_all = coords_all.detach().to(device=device, dtype=self.dtype).requires_grad_(True)
        self.model_type = model_type
        self.optimizer_type = optimizer_type.lower()
        self.tenkan_derivatives = tenkan_derivatives
        if tenkan_derivatives == "separable" and model_type != "TenKAN":
            raise ValueError("separable derivatives require the TenKAN model")
        self.lm_iter = None if lm_iter is None else int(lm_iter)
        self.k = int(k)
        self.n_interface_desired = int(n_interface_desired)
        self.tol = float(tol)
        self.adam_lr = float(adam_lr)
        self.rank = int(rank)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.kan_hidden = int(kan_hidden)
        self.kan_depth = int(kan_depth)
        self.seed = int(seed)
        self.interface_candidates = int(interface_candidates)
        self.test_points = int(test_points)
        self.eval_batch_size = int(eval_batch_size)
        self.output_root = os.path.abspath(
            output_root or os.path.join(current_dir, "results")
        )
        self.run_id = run_id
        self.save_plots = bool(save_plots)
        self.weight_bd = float(weight_bd)
        self.weight_f = float(weight_f)
        self.weight_jump = float(weight_jump)
        self.weight_normal_jump = float(weight_normal_jump)

        if self.rank <= 0 or self.grid_size <= 0 or self.spline_order < 0:
            raise ValueError("rank/grid-size must be positive and spline-order nonnegative")
        if self.interface_candidates <= 0 or self.test_points <= 0:
            raise ValueError("interface-candidates and test-points must be positive")

    def train(self):
        # 获取当前参数
        x_in = self.coords_in
        x_bc = self.coords_bc
        x_all = self.coords_all
        dtype = self.dtype

        # Gaussian mixture level-set φ(x)=c-Σ λ_i N(x;μ_i,Σ_i)
        def gm_phi(x):
            c = 0.3
            lambdas = [0.3, 0.3, 0.4]
            mus = [
                torch.tensor([-0.05, 0.25], dtype=x.dtype, device=x.device),
                torch.tensor([0.40, 0.30], dtype=x.dtype, device=x.device),
                torch.tensor([0.00,-0.35], dtype=x.dtype, device=x.device),
            ]
            Sigmas = [
                torch.tensor([[0.1, -0.08],[-0.08, 0.15]], dtype=x.dtype, device=x.device),
                torch.tensor([[0.1,  0.08],[ 0.08, 0.15]], dtype=x.dtype, device=x.device),
                torch.tensor([[0.15, 0.05],[0.05, 0.1]], dtype=x.dtype, device=x.device),
            ]
            total = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            for w, mu, S in zip(lambdas, mus, Sigmas):
                diff = x - mu
                invS = torch.inverse(S)
                detS = torch.det(S)
                quad = torch.einsum('bi,ij,bj->b', diff, invS, diff)
                norm = 1.0/(2.0*math.pi*torch.sqrt(detS))
                total = total + w * norm * torch.exp(-0.5*quad)
            return c - total

        phi = gm_phi(x_all)[:, None]
        phi_in = gm_phi(x_in)[:, None]

        # 界面采样并计算法向
        N_interface_desired = self.n_interface_desired
        tol = self.tol
        cand = sampling_points_test(N_inner=self.interface_candidates).to(
            device=device, dtype=dtype
        )
        phi_cand = gm_phi(cand)[:, None]
        band = (phi_cand.abs() < tol).squeeze()
        x_interface = cand[band]
        if x_interface.shape[0] > N_interface_desired:
            idx = torch.randperm(x_interface.shape[0], device=device)[:N_interface_desired]
            x_interface = x_interface[idx]
        x_interface = x_interface.detach().to(device).requires_grad_(True)
        phi_i = gm_phi(x_interface)[:, None]
        dphi = jacobian(phi_i, x_interface)
        normal = dphi / torch.norm(dphi, dim=1, keepdim=True)
        normal = normal.detach()

        # 准备训练数据（K 可调）
        k = self.k
        sign = sign_func(phi=phi_in)
        data = databuild(x_in, x_bc, x_interface, normal, sign, device=device, K=k)
        # 修正边界 g_D
        phi_bc = gm_phi(x_bc)[:, None]
        outside_bc = (phi_bc > 0).to(dtype)
        xb = x_bc[:,0:1]; yb = x_bc[:,1:2]
        g_bd = outside_bc*torch.cos(k*math.pi*xb)*torch.cos(k*math.pi*yb) + (1-outside_bc)*torch.sin(k*math.pi*xb)*torch.sin(k*math.pi*yb)
        data[1]["boundary"]["g_D"] = g_bd

        current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        default_run_id = (
            f"{current_date}_{self.model_type}_{self.optimizer_type}_"
            f"M{self.grid_size}_q{self.spline_order}_p{self.rank}_"
            f"K{self.k}_seed{self.seed}"
        )
        run_id = self.run_id or default_run_id
        if os.path.basename(run_id) != run_id or run_id in ("", ".", ".."):
            raise ValueError("run-id must be a non-empty file-name component")
        results_dir = os.path.join(self.output_root, run_id)
        completed_metrics = os.path.join(results_dir, "metrics.json")
        if os.path.exists(completed_metrics):
            raise FileExistsError(f"completed run already exists: {completed_metrics}")
        # A failed/preempted Slurm task may have left a partial directory. It is
        # safe to reuse it because the completion marker is metrics.json.
        os.makedirs(results_dir, exist_ok=True)

        # 模型与优化器
        model_type = self.model_type
        if model_type == 'KAN':
            model_u = KAN(
                [2, 8, 8, 2], grid_size=self.grid_size,
                spline_order=self.spline_order, base_activation=nn.Tanh
            ).to(device)
        elif model_type == 'TenKAN':
            model_u = KANProductSum(
                d=2, p=self.rank, hidden=self.kan_hidden,
                depth=self.kan_depth, out_features=2,
                grid_size=self.grid_size, spline_order=self.spline_order
            ).to(device)
        elif model_type == 'HRKAN':
            model_u = HRKAN([2,5,5,3,2], 5, 3, -1, 1, 4).to(device)
        elif model_type == 'MLP':
            model_u = Shallow_Deep_ext(2, 30, 2, layers=4,addiction_features=None).to(device)
        elif model_type == 'TNN':
            model_u = TNNProductSum(
                d=2, p=self.rank, hidden=20, depth=3, out_features=2
            ).to(device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        model_u_func_params = dict(model_u.named_parameters())
        # 打印模型的参数总数
        total_params = sum(p.numel() for p in model_u.parameters())
        print(f"模型的参数总数: {total_params}")

        optimizer_type = self.optimizer_type
        training_stats = {
            "actual_iterations": 0,
            "function_evaluations": None,
            "jacobian_evaluations": None,
            "final_loss": None,
        }
        training_status = "completed"
        train_start = None

        def begin_training_measurement():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            return time.perf_counter()

        if optimizer_type == 'lm':
            train_dic = prepare_train_state(
                model=model_u, model_func_params=model_u_func_params, data=data,
                tenkan_derivatives=self.tenkan_derivatives
            )
            LM_iter = self.lm_iter if self.lm_iter is not None else 1500
            train_start = begin_training_measurement()
            try:
                func_params_u, training_stats = lm_train(
                    train_dic, LM_iter=LM_iter, mu_update=3, div_factor=2,
                    mul_factor=3, continue_train=False, model_path=None,
                    return_stats=True
                )
                model_u_net = lambda x: functional_call(model_u, func_params_u, x)
            except KeyboardInterrupt:
                print("检测到手动中断(Ctrl-C)，提前结束 LM 训练并继续后续流程…")
                training_status = "interrupted"
                model_u_net = model_u
        elif optimizer_type == 'gdlm':
            train_dic = prepare_train_state(model=model_u, model_func_params=model_u_func_params,
                                           data=data, tenkan_derivatives=self.tenkan_derivatives)
            LM_iter = self.lm_iter if self.lm_iter is not None else 1600
            train_start = begin_training_measurement()
            try:
                func_params_u, _, training_stats = gdlm_train(
                    train_dic, LM_iter=LM_iter, imethod=2, iaccel=1,
                    ibold=2, continue_train=False, model_path=None,
                    otherkw={}, return_stats=True
                )
                model_u_net = lambda x: functional_call(model_u, func_params_u, x)
            except KeyboardInterrupt:
                print("检测到手动中断(Ctrl-C)，提前结束 GdLM 训练并继续后续流程…")
                training_status = "interrupted"
                model_u_net = model_u
        elif optimizer_type in ('lbfgs', 'adam'):
            # 构建与 ex17 一致的单模型训练生成器（使用数据中的 Beta/Gradbeta）
            Beta = data[1]["inner"]["beta"].to(dtype).to(device)
            Gradbeta = data[1]["inner"]["grad_beta"].to(dtype).to(device)
            iBeta1 = data[1]["interface"]["ibeta1"].to(dtype).to(device)
            iBeta2 = data[1]["interface"]["ibeta2"].to(dtype).to(device)

            X_bd = data[0]["boundary"].to(dtype).to(device)
            U_bd = data[1]["boundary"]["g_D"].to(dtype).to(device)

            X_in = data[0]["inner"].to(dtype).to(device)
            U_f = data[1]["inner"]["f"].to(dtype).to(device)
            Sign = data[1]["inner"]["sign"].to(dtype).to(device)

            X_interf = data[0]["interface"].to(dtype).to(device)
            U_w = data[1]["interface"]["w"].to(dtype).to(device)
            Normal_ij = data[1]["interface"]["normal_vector"].to(dtype).to(device)
            U_v = data[1]["interface"]["v"].to(dtype).to(device)

            mse_loss = torch.nn.MSELoss(reduction="mean")

            def loss_boundary_coupling(net, X_bd, U_bd, weight_bd):
                out = net(X_bd)
                U_pred = out[:, 0:1]
                return mse_loss(U_pred, U_bd) * weight_bd

            def loss_f_coupling(net, X_in, U_f, Sign, Beta, Gradbeta, weight_f):
                if self.tenkan_derivatives == "separable":
                    _, gradient, laplacian = net(X_in, derivative_order=2)
                    grad_y = gradient[:, 0, :] + Sign * gradient[:, 1, :]
                    lap = laplacian[:, 0:1] + Sign * laplacian[:, 1:2]
                else:
                    out = net(X_in)
                    y = out[:, 0:1] + Sign * out[:, 1:2]
                    grad_y = torch.autograd.grad(
                        y, X_in, grad_outputs=torch.ones_like(y), create_graph=True
                    )[0]
                    lap = 0.0
                    for d in range(X_in.shape[1]):
                        gk = grad_y[:, d:d+1]
                        d2_k = torch.autograd.grad(
                            gk, X_in, grad_outputs=torch.ones_like(gk), create_graph=True
                        )[0][:, d:d+1]
                        lap = lap + d2_k
                lhs = -lap * Beta - (grad_y * Gradbeta).sum(dim=1, keepdim=True)
                return mse_loss(lhs, U_f) * weight_f

            def loss_jump_coupling(net, X_interf, U_w, weight_jump):
                return mse_loss(net(X_interf)[:, 1:2], -U_w) * weight_jump

            def loss_normal_jump_coupling(net, X_interf, Normal_ij, U_v, iBeta1, iBeta2, weight_normal_jump):
                if self.tenkan_derivatives == "separable":
                    _, gradient, _ = net(X_interf, derivative_order=1)
                    grad_U = gradient[:, 0, :]
                    grad_V = gradient[:, 1, :]
                else:
                    out = net(X_interf)
                    U_pred = out[:, 0:1]
                    V_pred = out[:, 1:2]
                    grad_U = torch.autograd.grad(
                        U_pred, X_interf, grad_outputs=torch.ones_like(U_pred), create_graph=True
                    )[0]
                    grad_V = torch.autograd.grad(
                        V_pred, X_interf, grad_outputs=torch.ones_like(V_pred), create_graph=True
                    )[0]
                pred_nj = (((iBeta2 - iBeta1) * grad_U - iBeta1 * grad_V) * Normal_ij).sum(dim=1, keepdim=True)
                return mse_loss(pred_nj, U_v) * weight_normal_jump

            weight_bd = self.weight_bd
            weight_f = self.weight_f
            weight_jump = self.weight_jump
            weight_normal_jump = self.weight_normal_jump

            def generator_flat(full_batch=True):
                while True:
                    yield {
                        "bd": (loss_boundary_coupling, (X_bd, U_bd, weight_bd)),
                        "f": (loss_f_coupling, (X_in, U_f, Sign, Beta, Gradbeta, weight_f)),
                        "jump": (loss_jump_coupling, (X_interf, U_w, weight_jump)),
                        "normal_jump": (loss_normal_jump_coupling, (X_interf, Normal_ij, U_v, iBeta1, iBeta2, weight_normal_jump)),
                    }

            if optimizer_type == 'lbfgs':
                LM_iter = self.lm_iter if self.lm_iter is not None else 20000
                train_start = begin_training_measurement()
                try:
                    _, training_stats = lbfgs_train(
                        generator_flat(full_batch=True), model_u,
                        max_iter=LM_iter, model_path=None,
                        return_stats=True
                    )
                except KeyboardInterrupt:
                    print("检测到手动中断(Ctrl-C)，提前结束 LBFGS 训练并继续后续流程…")
                    training_status = "interrupted"
                model_u_net = model_u
            else:  # adam
                LM_iter = self.lm_iter if self.lm_iter is not None else 410
                train_start = begin_training_measurement()
                try:
                    _, training_stats = adam_train(
                        generator_flat(full_batch=True), model_u,
                        epoch=LM_iter, learning_rate=self.adam_lr,
                        model_path=None, return_stats=True
                    )
                except KeyboardInterrupt:
                    print("检测到手动中断(Ctrl-C)，提前结束 Adam 训练并继续后续流程…")
                    training_status = "interrupted"
                model_u_net = model_u
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        usetime = time.perf_counter() - train_start
        peak_gpu_memory_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None
        )

        # 测试与评估。L2/L∞ 与 broken H1 使用同一组确定性均匀网格点。
        x_test = sampling_points_linspace_test(N_inner=self.test_points).to(
            device=device, dtype=dtype
        )
        with torch.no_grad():
            u_out = model_u_net(x_test).detach()
        phi_test = gm_phi(x_test)[:, None]
        sign_x = sign_func(phi=phi_test)
        u = u_out[:,0:1] + sign_x*u_out[:,1:2]

        linf_error, rel_linf_error, l2_error, rel_l2_error, l2_error_inside, l2_error_outside, l2_error_interface, point_errors = compute_errors_anly(x_test, u, phi_test, K=k)
        broken_h1 = compute_broken_h1_anly(
            model_u_net, x_test, phi_test, K=k,
            batch_size=self.eval_batch_size
        )

        print("\n--------  误差评估 --------")
        print(f"L2误差: {l2_error.item():.16f}")
        print(f"相对L2误差: {rel_l2_error.item():.16f}")
        print(f"L∞误差: {linf_error.item():.16f}")
        print(f"相对L∞误差: {rel_linf_error.item():.16f}")
        print(f"界面内部L2误差: {l2_error_inside.item():.16f}")
        print(f"界面外部L2误差: {l2_error_outside.item():.16f}")
        if not torch.isnan(l2_error_interface):
            print(f"界面附近L2误差: {l2_error_interface.item():.16f}")
        print(f"相对 broken H1 误差: {broken_h1['relative_broken_h1']:.16f}")

        actual_iterations = int(training_stats.get("actual_iterations") or 0)
        time_per_iteration = (
            usetime / actual_iterations if actual_iterations > 0 else None
        )
        peak_gpu_memory_mib = (
            peak_gpu_memory_bytes / (1024.0 ** 2)
            if peak_gpu_memory_bytes is not None else None
        )

        def scalar(value):
            if value is None:
                return None
            if torch.is_tensor(value):
                value = value.detach().cpu().item()
            value = float(value)
            return value if math.isfinite(value) else None

        metrics = {
            "schema_version": 1,
            "run_id": run_id,
            "status": training_status,
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "example": "Example 4.2 Gaussian-mixture interface",
            "model_type": model_type,
            "optimizer_type": optimizer_type,
            "optimizer_implementation": {
                "adam": "torch.optim.AdamW",
                "lbfgs": "torch.optim.LBFGS",
                "gdlm": "geodesiclm",
                "lm": "custom LM",
            }[optimizer_type],
            "tenkan_derivatives": self.tenkan_derivatives,
            "iterations_requested": int(LM_iter),
            "requested_iteration_unit": "epoch" if optimizer_type == "adam" else "optimizer iteration",
            "iterations_actual": actual_iterations,
            "actual_iteration_unit": "parameter update" if optimizer_type == "adam" else "optimizer iteration",
            "function_evaluations": training_stats.get("function_evaluations"),
            "jacobian_evaluations": training_stats.get("jacobian_evaluations"),
            "training_time_seconds": usetime,
            "time_per_iteration_seconds": time_per_iteration,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "peak_gpu_memory_mib": peak_gpu_memory_mib,
            "l2_error": scalar(l2_error),
            "relative_l2_error": scalar(rel_l2_error),
            "linf_error": scalar(linf_error),
            "relative_linf_error": scalar(rel_linf_error),
            "broken_h1_error": broken_h1["broken_h1_rms"],
            "relative_broken_h1_error": broken_h1["relative_broken_h1"],
            "broken_h1_seminorm_error": broken_h1["broken_h1_seminorm_rms"],
            "relative_broken_h1_seminorm_error": broken_h1["relative_broken_h1_seminorm"],
            "l2_error_inside": scalar(l2_error_inside),
            "l2_error_outside": scalar(l2_error_outside),
            "l2_error_interface_band": scalar(l2_error_interface),
            "rank": self.rank,
            "p": self.rank,
            "spline_order": self.spline_order,
            "q": self.spline_order,
            "grid_size": self.grid_size,
            "M": self.grid_size,
            "grid_and_spline_parameters_active": model_type in ("KAN", "TenKAN"),
            "rank_parameter_active": model_type in ("TenKAN", "TNN"),
            "K": k,
            "seed": self.seed,
            "kan_hidden": self.kan_hidden,
            "kan_depth": self.kan_depth,
            "trainable_parameters": total_params,
            "n_inner": len(self.coords_in),
            "n_boundary": len(self.coords_bc),
            "n_interface_requested": self.n_interface_desired,
            "n_interface_actual": len(x_interface),
            "interface_candidates": self.interface_candidates,
            "interface_band_tolerance": self.tol,
            "test_points_requested": self.test_points,
            "test_points_actual": len(x_test),
            "broken_h1_test_points": broken_h1["broken_h1_test_points"],
            "broken_h1_interface_points_excluded": broken_h1["excluded_interface_points"],
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "adam_learning_rate": self.adam_lr if optimizer_type == "adam" else None,
            "adam_weight_decay": 0.01 if optimizer_type == "adam" else None,
            "adam_updates_per_epoch": 50 if optimizer_type == "adam" else None,
            "adam_scheduler_step_size_epochs": 50 if optimizer_type == "adam" else None,
            "adam_scheduler_gamma": 0.7 if optimizer_type == "adam" else None,
            "lbfgs_learning_rate": 1.0 if optimizer_type == "lbfgs" else None,
            "lbfgs_history_size": 100 if optimizer_type == "lbfgs" else None,
            "lbfgs_tolerance_grad": 1e-9 if optimizer_type == "lbfgs" else None,
            "lbfgs_tolerance_change": 0.0 if optimizer_type == "lbfgs" else None,
            "lbfgs_line_search": "strong_wolfe" if optimizer_type == "lbfgs" else None,
            "gdlm_imethod": 2 if optimizer_type == "gdlm" else None,
            "gdlm_iaccel": 1 if optimizer_type == "gdlm" else None,
            "gdlm_ibold": 2 if optimizer_type == "gdlm" else None,
            "gdlm_frtol": 1e-5 if optimizer_type == "gdlm" else None,
            "gdlm_avmax": 0.1 if optimizer_type == "gdlm" else None,
            "loss_weighting": (
                "weighted mean-square components"
                if optimizer_type in ("adam", "lbfgs")
                else "per-component residual L2 normalization"
            ),
            "weight_boundary": self.weight_bd if optimizer_type in ("adam", "lbfgs") else None,
            "weight_residual": self.weight_f if optimizer_type in ("adam", "lbfgs") else None,
            "weight_jump": self.weight_jump if optimizer_type in ("adam", "lbfgs") else None,
            "weight_normal_jump": self.weight_normal_jump if optimizer_type in ("adam", "lbfgs") else None,
            "final_training_loss": scalar(training_stats.get("final_loss")),
        }

        with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        with open(os.path.join(results_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics))
            writer.writeheader()
            writer.writerow(metrics)

        # Model serialization is deliberately outside the timed training phase.
        if optimizer_type in ("lm", "gdlm") and "func_params_u" in locals():
            torch.save(
                {name: value.detach().cpu() for name, value in func_params_u.items()},
                os.path.join(results_dir, "model.pt")
            )
        elif optimizer_type in ("adam", "lbfgs"):
            torch.save(
                {name: value.detach().cpu() for name, value in model_u.state_dict().items()},
                os.path.join(results_dir, "model.pt")
            )

        with open(os.path.join(results_dir, "error_metrics.txt"), "w", encoding="utf-8") as f:
            f.write("-------- 训练参数 --------\n")
            f.write(f"训练时间:{usetime}s\n")
            f.write(f"网络模型:{model_type}\n")
            f.write(f"方程参数k值:{k}\n")
            f.write(f"模型的参数总数: {total_params}\n")
            f.write(f"内部采样点个数：{len(self.coords_in)}\n")
            f.write(f"边界采样点个数：{len(self.coords_bc)}\n")
            f.write(f"界面采样点个数：{len(x_interface)}\n")
            f.write(f"优化器类型: {optimizer_type}\n")
            f.write(f"请求迭代预算: {LM_iter}\n")
            f.write(f"实际迭代次数: {actual_iterations}\n")
            f.write(f"每次迭代时间: {time_per_iteration}s\n")
            f.write(f"峰值GPU显存: {peak_gpu_memory_mib} MiB\n")
            f.write(f"M/grid size: {self.grid_size}\n")
            f.write(f"q/spline order: {self.spline_order}\n")
            f.write(f"p/rank: {self.rank}\n")
            f.write(f"seed: {self.seed}\n")
            f.write("-------- 误差评估 --------\n")
            f.write(f"L2误差: {l2_error.item():.16f}\n")
            f.write(f"相对L2误差: {rel_l2_error.item():.16f}\n")
            f.write(f"L∞误差: {linf_error.item():.16f}\n")
            f.write(f"相对L∞误差: {rel_linf_error.item():.16f}\n")
            f.write(f"界面内部L2误差: {l2_error_inside.item():.16f}\n")
            f.write(f"界面外部L2误差: {l2_error_outside.item():.16f}\n")
            if not torch.isnan(l2_error_interface):
                f.write(f"界面附近L2误差: {l2_error_interface.item():.16f}\n")
            f.write(f"broken H1误差: {broken_h1['broken_h1_rms']:.16f}\n")
            f.write(f"相对broken H1误差: {broken_h1['relative_broken_h1']:.16f}\n")

        if self.save_plots:
            fig = visualize_data(xy=x_test, u=point_errors, mode='2d', show=False)
            fig.savefig(os.path.join(results_dir, "01_abs_error_distribution.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
            fig = visualize_data(xy=x_test, u=u, mode='3d', show=False)
            fig.savefig(os.path.join(results_dir, "02_u_3d.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
            fig = visualize_data(xy=x_test, u=u, mode='2d', show=False)
            fig.savefig(os.path.join(results_dir, "02_u_2d.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
        print(f"\n结果已保存至: {results_dir}")
        print(f"METRICS_JSON={os.path.join(results_dir, 'metrics.json')}")

        return metrics


# 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Gaussian mixture interface PINN training")
    parser.add_argument('--n-inner', type=int, default=2100, help='内部采样点数')
    parser.add_argument('--n-boundary', type=int, default=160, help='边界采样点总数（自动均分到四条边）')
    parser.add_argument('--n-interface', type=int, default=160, help='界面目标采样点数')
    parser.add_argument('--tol', type=float, default=5e-3, help='界面采样带宽 |phi|<tol')
    parser.add_argument('--model-type', type=str, default='TenKAN', choices=['KAN','TenKAN','HRKAN','MLP','TNN'], help='网络模型类型')
    parser.add_argument('--optimizer-type', type=str, default='gdlm', choices=['adam','gdlm','lm','lbfgs'], help='优化器类型')
    parser.add_argument('--lm-iter', '--iterations', dest='lm_iter', type=int, default=None,
                        help='优化预算；默认 Adam=410 epoch、L-BFGS=20000、GdLM=1600、LM=1500')
    parser.add_argument('--adam-lr', type=float, default=5e-3, help='Adam 学习率')
    parser.add_argument('--k', '--K', dest='k', type=int, default=5, help='方程频率参数 K')
    parser.add_argument('--grid-size', '--M', dest='grid_size', type=int, default=15,
                        help='KAN B-spline grid size M')
    parser.add_argument('--spline-order', '--q', dest='spline_order', type=int, default=6,
                        help='B-spline order q')
    parser.add_argument('--rank', '--p', dest='rank', type=int, default=5,
                        help='TenKAN/TNN separation rank p')
    parser.add_argument('--kan-hidden', type=int, default=5, help='每个一维 KAN 子网络的隐藏宽度')
    parser.add_argument('--kan-depth', type=int, default=2, help='每个一维 KAN 子网络的隐藏层数')
    parser.add_argument('--tenkan-derivatives', choices=['autograd', 'separable'],
                        default='autograd', help='TenKAN spatial derivative backend for all optimizers')
    parser.add_argument('--dtype', type=str, default='float64', help='浮点精度类型：float32 或 float64')
    parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    parser.add_argument('--interface-candidates', type=int, default=200000,
                        help='用于筛选界面点的候选点数')
    parser.add_argument('--test-points', type=int, default=90000,
                        help='误差评估点数（实际取 floor(sqrt(N))^2）')
    parser.add_argument('--eval-batch-size', type=int, default=8192,
                        help='broken H1 自动微分评估批大小')
    parser.add_argument('--output-root', type=str,
                        default=os.path.join(current_dir, 'results'),
                        help='每个 run 子目录的根目录')
    parser.add_argument('--run-id', type=str, default=None,
                        help='可选的唯一运行标识；Slurm sweep 会自动提供')
    parser.add_argument('--no-plots', action='store_true',
                        help='跳过单次运行的解/误差图（不影响汇总收敛图）')
    # 新增：损失权重参数
    parser.add_argument('--weight-bd', type=float, default=1.0, help='边界条件损失权重')
    parser.add_argument('--weight-f', type=float, default=1.0, help='方程残差损失权重')
    parser.add_argument('--weight-jump', type=float, default=1.0, help='界面跳跃条件损失权重')
    parser.add_argument('--weight-normal-jump', type=float, default=1.0, help='法向通量跳跃损失权重')
    args = parser.parse_args()

    # dtype
    dtype_map = {'float32': torch.float32, 'float64': torch.float64}
    dtype_str = getattr(args, 'dtype', 'float64').lower()
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype argument: {args.dtype}. Use float32 or float64.")
    selected_dtype = dtype_map[dtype_str]
    torch.set_default_dtype(selected_dtype)

    # seed
    seed = int(args.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # 采样：边界和内部点（保持与原脚本一致的策略）
    _, coords_bc, _ = sampling_points_random(N_inner=0, N_boundary=args.n_boundary)
    coords_in_np = random_bbox_sampler(n=args.n_inner, dim=2, bbox=np.array([[-1, -1], [1, 1]]), random="Hammersley")
    coords_in = torch.tensor(coords_in_np, dtype=selected_dtype).to(device)
    coords_all = torch.cat([coords_in, coords_bc.to(dtype=selected_dtype)], dim=0)

    trainer = Trainer(
        coords_in=coords_in,
        coords_bc=coords_bc,
        coords_all=coords_all,
        model_type=args.model_type,
        optimizer_type=args.optimizer_type,
        lm_iter=args.lm_iter,
        k=args.k,
        n_interface_desired=args.n_interface,
        tol=args.tol,
        dtype=selected_dtype,
        adam_lr=args.adam_lr,
        rank=args.rank,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
        kan_hidden=args.kan_hidden,
        kan_depth=args.kan_depth,
        tenkan_derivatives=args.tenkan_derivatives,
        seed=args.seed,
        interface_candidates=args.interface_candidates,
        test_points=args.test_points,
        eval_batch_size=args.eval_batch_size,
        output_root=args.output_root,
        run_id=args.run_id,
        save_plots=not args.no_plots,
        weight_bd=args.weight_bd,
        weight_f=args.weight_f,
        weight_jump=args.weight_jump,
        weight_normal_jump=args.weight_normal_jump,
    )

    trainer.train()
