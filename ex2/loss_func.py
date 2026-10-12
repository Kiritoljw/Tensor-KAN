# from functorch import vmap, grad, jacrev, make_functional
import torch
import sys
import os
import numpy as np

# 添加父目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from torch.func import vmap,grad,jacrev,functional_call
from module.gradient import jacobian,hessian
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

#===============================================
# loss for co-training LM优化器损失函数
#===============================================

#仅用于model_u的损失函数
def loss_bd_lm(func_params, X_bd, U_bd):
    def f(x, func_params):
        #print(f"X_bd: {X_bd.shape},U_bd: {U_bd.shape}, ")
        output = functional_call(loss_bd_lm._model,func_params, x)

        return output[0], output[1]
    u_pred , _ = f(X_bd, func_params)
    loss_bd= u_pred - U_bd
    loss_bd = loss_bd
    return loss_bd.flatten()

def loss_jump_lm(func_params, X_interf, U_w):
    def f(x, func_params):
        #print(f"X_bd: {X_interf.shape},U_bd: {U_w.shape}, ")
        output = functional_call(loss_jump_lm._model,func_params, x)
        return output[0], output[1]
    _ , V_pred = f(X_interf, func_params)
    loss_jump = - V_pred - U_w
    return loss_jump.flatten()

#联合损失函数
def loss_normal_jump_lm_joint(func_params, X_ij, Normal_ij, Unj_ij,iBeta1,iBeta2):
    """
    ((ibeta2-ibeta1)*div(U)- ibeta1*div(V))*normal-u_nj
    """
    def f(x, func_params):
        #print(f"X_ij: {X_ij.shape},Un_ij: {Unj_ij.shape}, ")
        output = functional_call(loss_normal_jump_lm_joint._model,func_params, x)
        return output[0], output[1]

    grad_U , grad_V = jacrev(f)(X_ij, func_params)
    loss_normal_jump = ((iBeta2-iBeta1)*grad_U-iBeta1*grad_V).dot(Normal_ij) - Unj_ij
    return loss_normal_jump.flatten()

def loss_f_lm_joint(func_params, X_in, U_f, Sign, Beta, Gradbeta):
    def f(x, func_params):
        #print(f"X_in: {X_in.shape},U_bd: {U_f.shape},Sign:{Sign.shape} ")
        output = functional_call(loss_f_lm_joint._model,func_params, x)
        return (output[0]+ Sign * output[1]).squeeze(0)

    u = f(X_in,func_params)
    d1u = grad(f)(X_in, func_params)   # jacobian of u
    d2u = jacrev(grad(f))(X_in, func_params) # hessian of u
    dudX2 = torch.diagonal(d2u)

    # -div(beta grad u) = -beta * Δu - grad(beta)·grad(u)
    laplacian_term = -Beta*torch.sum(dudX2)
    grad_beta_term = -(d1u*Gradbeta).sum()
    lhs = laplacian_term + grad_beta_term
    loss_f_inhom_vc = lhs - U_f
    return loss_f_inhom_vc.flatten()

# 更新损失函数列表
loss_list = [getattr(sys.modules[__name__], name) for name in dir()
             if name.startswith("loss_") ]

# 设置模型
def setup_fm_dual(model):
    for loss in loss_list:
        loss._model = model

def prepare_train_state(model,model_func_params,data, tenkan_derivatives="autograd"):
    # set up ._func_model for compute_loss functions
    #use for LM优化器
    setup_fm_dual(model)

    x_dict,data_dict=data
    reference_parameter = next(iter(model_func_params.values()))
    target_dtype = reference_parameter.dtype
    target_device = reference_parameter.device

    def convert(value):
        return value.to(device=target_device, dtype=target_dtype)

    X_bd=convert(x_dict["boundary"])
    U_bd=convert(data_dict["boundary"]["g_D"])

    X_in=convert(x_dict["inner"])
    U_f=convert(data_dict["inner"]["f"])
    Sign=convert(data_dict["inner"]["sign"])
    Beta=convert(data_dict["inner"]["beta"])
    grad_beta=convert(data_dict["inner"]["grad_beta"])

    X_interf=convert(x_dict["interface"])
    U_w=convert(data_dict["interface"]["w"])
    Normal_ij=convert(data_dict["interface"]["normal_vector"])
    U_v=convert(data_dict["interface"]["v"])
    iBeta1=convert(data_dict["interface"]["ibeta1"])
    iBeta2=convert(data_dict["interface"]["ibeta2"])


    train_dic={"bd":(loss_bd_lm, (None, 0, 0), (model_func_params, X_bd, U_bd)),
            "jump": (loss_jump_lm, (None, 0, 0), (model_func_params, X_interf, U_w)),
            "normal_jump":(loss_normal_jump_lm_joint, (None, 0, 0, 0, 0, 0), (model_func_params, X_interf, Normal_ij, U_v, iBeta1, iBeta2)),
        "f":(loss_f_lm_joint, (None, 0, 0, 0, 0, 0), (model_func_params, X_in, U_f, Sign, Beta, grad_beta)),
    }

    if tenkan_derivatives == "separable":
        from module.kan import KANProductSum
        if not isinstance(model, KANProductSum) or model.l2_normalize:
            raise ValueError("separable derivatives require an unnormalised TenKAN")

        def fast_flux(params, point, normal, target, beta1, beta2):
            _, gradient, _ = functional_call(
                model, params, (point,), {"derivative_order": 1})
            pred = ((beta2 - beta1) * gradient[0] - beta1 * gradient[1]).dot(normal)
            return (pred - target).flatten()

        def fast_pde(params, point, target, indicator, beta, grad_beta):
            _, gradient, laplacian = functional_call(
                model, params, (point,), {"derivative_order": 2})
            du = gradient[0] + indicator * gradient[1]
            lap = laplacian[0] + indicator * laplacian[1]
            return (-beta * lap - (du * grad_beta).sum() - target).flatten()

        for name, function in (("normal_jump", fast_flux), ("f", fast_pde)):
            _, axes, arguments = train_dic[name]
            train_dic[name] = (function, axes, arguments)
    elif tenkan_derivatives != "autograd":
        raise ValueError("Unknown TenKAN derivative backend")
    return train_dic

#===============================================
# loss for co-training 正常优化器损失函数
#===============================================
mse_loss = torch.nn.MSELoss(reduction="mean")
def state_loss_boundary_coupling(net, X_bd, U_bd, weight_bd):
    out = net(X_bd)
    U_pred = out[:,0:1]
    return mse_loss(U_pred,U_bd)*weight_bd

def state_loss_f_coupling(net,net_beta, X_in, U_f, Sign, weight_f):
    out = net(X_in)
    b_out = net_beta(X_in)
    y=out[:,0:1]+Sign*out[:,1:2]
    Beta = b_out[:,0:1]+Sign*b_out[:,1:2]
    grad_y= jacobian(y,X_in)
    lhs =0 # left hand side
    for k in range(X_in.shape[1]):
        lhs = lhs + hessian(y,X_in,i=k,j=k,grad_y=grad_y)
    lhs= - lhs*Beta #- (grad_y*Gradbeta).sum(dim=1,keepdim=True)  # (batch, 1)

    return mse_loss(lhs, U_f)*weight_f

def state_loss_jump_coupling(net, X_interf, U_w):
    return mse_loss(-net(X_interf)[:,1:2],U_w)

def state_loss_normal_jump_coupling(net,net_beta, X_interf, Normal_ij, U_v):
    out = net(X_interf)
    U_pred=out[:,0:1]
    V_pred=out[:,1:2]
    beta_pred=net_beta(X_interf)
    iBeta1=beta_pred[:,0:1]+beta_pred[:,1:2]
    iBeta2=beta_pred[:,0:1]
    grad_U=jacobian(U_pred,X_interf)
    grad_V=jacobian(V_pred,X_interf)
    pred_nj = (((iBeta2-iBeta1)*grad_U-iBeta1*grad_V)*Normal_ij).sum(dim=1,keepdim=True)#((-beta_pred*grad_V)*Normal_ij).sum(dim=1,keepdim=True)
    return mse_loss(pred_nj,U_v)

def state_loss_data(net,X_data,U_data,Sign_data,weight_data):
    out = net(X_data)
    u_pred = out[:,0:1]+Sign_data*out[:,1:2]
    return mse_loss(u_pred,U_data) * weight_data

def regularization_loss(net,X,Sign,weight_BF):
    """L2-regularization"""
    b_out = net(X)
    Beta = b_out[:,0:1]+Sign*b_out[:,1:2]
    L2_norm_sq = torch.sum(Beta**2)
    loss = L2_norm_sq * weight_BF

    return loss


def gen_train_dic(data, full_batch=True, device=device):
    # use fixed sampling points for adam or lbfgs优化

    x_dict,data_dict=data

    X_data=x_dict["x_data"].to(torch.float64).to(device)
    U_data=data_dict["data"]["u_data"].to(torch.float64).to(device)
    Sign_data=data_dict["data"]["sign_data"].to(torch.float64).to(device)

    X_bd=x_dict["boundary"].to(torch.float64).to(device)
    U_bd=data_dict["boundary"]["g_D"].to(torch.float64).to(device)

    X_in=x_dict["inner"].to(torch.float64).to(device)
    U_f=data_dict["inner"]["f"].to(torch.float64).to(device)
    Sign=data_dict["inner"]["sign"].to(torch.float64).to(device)

    X_interf=x_dict["interface"].to(torch.float64).to(device)
    U_w=data_dict["interface"]["w"].to(torch.float64).to(device)
    Normal_ij=data_dict["interface"]["normal_vector"].to(torch.float64).to(device)
    U_v=data_dict["interface"]["v"].to(torch.float64).to(device)

    X_regular=x_dict["regularization"].to(torch.float64).to(device)
    Sign_regular=data_dict["regularization"].to(torch.float64).to(device)

    # iBeta1=data_dict["interface"]["ibeta1"].to(torch.float64)
    # iBeta2=data_dict["interface"]["ibeta2"].to(torch.float64)

    # only for adam or lbfgs
    weight_bd = 100
    weight_f = 1
    weight_data = 1
    weight_regular = 0.00001

    if full_batch:
        # full-batch mode
        while True:
            train_dic={}
            #只需要u模型的损失函数
            train_dic["model_u"] = {
                "bd": (state_loss_boundary_coupling, (X_bd, U_bd, weight_bd)),
                "jump": (state_loss_jump_coupling, (X_interf, U_w)),
                "data_loss": (state_loss_data, (X_data, U_data, Sign_data, weight_data))
            }

            train_dic["joint"] = {
                "f": (state_loss_f_coupling, (X_in, U_f, Sign, weight_f)),
                "normal_jump": (state_loss_normal_jump_coupling, (X_interf, Normal_ij, U_v))
            }
            train_dic["model_beta"]={
                "regularization_loss":(regularization_loss,(X_regular,Sign_regular,weight_regular))
            }
            yield train_dic
            # train_dic["bd"]= (state_loss_boundary_coupling,(X_bd,U_bd, weight_bd))
            # train_dic["f"]= (state_loss_f_coupling,(X_in,U_f, Sign,  weight_f))
            # train_dic["jump"]= (state_loss_jump_coupling,(X_interf, U_w))
            # train_dic["normal_jump"]= (state_loss_normal_jump_coupling,(X_interf, Normal_ij,U_v))
            # train_dic["data_loss"]=(state_loss_data),(X_data,U_data,Sign_data,weight_data)
            # yield train_dic
