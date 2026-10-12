import torch
import numpy as np

from torch.func import vmap,grad,jacrev,functional_call
from .gradient import caches_clear  # 为PINN梯度/ Hessian缓存清理添加导入，避免跨iteration复用已释放计算图
import os
import time
from .utils import params_pack,params_unpack,clear_cuda_cache
# device = torch.device('cpu')
# torch.set_default_dtype(torch.float64)

from .geodesiclm_v1 import geodesiclm

# Helper: ensure parent directory exists for a checkpoint/model path
def _ensure_parent_dir(path: str | None):
    if path:
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

# standard LM
def lm_train(train_dic, LM_iter=10000,mu_update=3,div_factor = 2 ,mul_factor = 3,
             continue_train=False, model_path=None, return_stats=False):
    """
    train_dic: {name: (func,vm_args,func_args),...}, use to call vmap(func,vm_args)(func_args).
                    See the usage of torch.func.vmap for more information.
    mu_update : update \mu every mu_update iterations
    div_factor : \mu <- \mu/div_factor when loss decreases
    mul_factor : \mu <- mul_factor*\mu when loss incerases
    """

    func_params=next(iter(train_dic.values()))[2][0]
    x0,signature=params_unpack(func_params)
    if continue_train:
        assert os.path.exists(model_path), f"model_path{model_path} doesn't exist!"
        checkpoint=torch.load(model_path)
        mu = checkpoint["mu"]
        loss_sum_old = checkpoint["loss_sum_old"]
        itera = checkpoint["itera"]
        history = checkpoint["history"]
        func_params = checkpoint["func_params"]
        x0,signature=params_unpack(func_params)
    else:
        mu = torch.tensor(10**5).to(x0)
        loss_sum_old = 10**10
        itera = 0
        history = []
    mul_factor = torch.tensor(mul_factor).to(x0)
    div_factor = torch.tensor(div_factor).to(x0)
    I = torch.eye(x0.shape[0]).to(x0)
    x = x0
    time1=time.time()
    count=0
    steps_run = 0
    for step in range(LM_iter+1):
        steps_run += 1
        # Put into loss functional to get L_vec
        L_vec=[]
        J_mat=[]
        func_params_=params_pack(x,signature)
        for loss_func,vm_args,func_args in train_dic.values():
            L_item=vmap(loss_func,vm_args)(func_params_,*func_args[1:])
            L_item/=np.sqrt(len(L_item))
            L_vec.append(L_item)
            per_sample_grads = vmap(jacrev(loss_func), vm_args)(func_params_,*func_args[1:])
            cnt = 0
            for g in per_sample_grads.values():
                g = g.detach()
                J_item = g.view(len(g), -1) if cnt == 0 else torch.hstack([J_item, g.view(len(g), -1)])
                cnt = 1
            J_mat.append(J_item)

        J_mat=torch.cat(J_mat)
        L_vec=torch.cat(L_vec)
        with torch.no_grad():
            #compute loss
            loss=0
            for vec in L_vec:
                loss = loss + torch.sum(vec**2)
            history.append(loss.item())
            #update parameters
            J_product = J_mat.t()@J_mat
            rhs = -J_mat.t()@L_vec
            # dp = torch.linalg.solve(J_product + mu*I, rhs)
            L, info= torch.linalg.cholesky_ex(J_product + mu*I)  #returning the LAPACK error codes, which is faster than torch.linalg.cholesky
            if info==0:
                dp = torch.linalg.solve_triangular(L.t(),torch.linalg.solve_triangular(L,rhs,upper=False),upper=True).squeeze()
            else:  # reject
                mu=mu*mul_factor
                itera+=1
                continue
            x_new=x+dp

            # reject if loss increases
            loss_new=0
            func_params_=params_pack(x_new,signature)
            for loss_func,vm_args,func_args in train_dic.values():
                L_item=vmap(loss_func,vm_args)(func_params_,*func_args[1:])
                L_item/=np.sqrt(len(L_item))
                loss_new = loss_new + torch.sum(L_item**2)
            if loss_new > loss: # reject
                mu=mu*mul_factor
                itera+=1
                continue
            else:
                x=x_new



        itera += 1
        if step % mu_update == 0:
            #if loss_sum_check < loss_sum_old:
            if loss < loss_sum_old:
                mu = max(mu/div_factor, 10**(-15))
                count=0
            else:
                mu = min(mul_factor*mu, 10**(3))
                if mu == 10**(3):
                    count+=1
                if count>=5:
                    print("lm algorithm bloomed up,Iter %d, Loss_Res: %.5e, mu: %.5e" % (itera, loss.item(), mu) )
                    break
            loss_sum_old = loss

        if step%100 == 0:
            print(
                    'Iter %d, Loss_Res: %.5e, mu: %.5e' % (itera, loss.item(), mu)
                )
        if step% 1000 == 0 and model_path:
            func_params_=params_pack(x,signature)
            _ensure_parent_dir(model_path)
            torch.save({"mu":mu, "loss_sum_old":loss_sum_old, "itera":itera, "func_params": func_params, "history": history}, model_path)
        if step == LM_iter or loss.item()<10**(-15):
                break
    # save results
    time2=time.time()
    func_params=params_pack(x,signature)
    if model_path:
        _ensure_parent_dir(model_path)
        torch.save({"mu":mu, "loss_sum_old":loss_sum_old, "itera":itera, "func_params": func_params, "history": history}, model_path)
        print(f"Trained results saved at {model_path}. Using time {time2-time1:.2f}s")
    stats = {
        "actual_iterations": steps_run,
        "function_evaluations": None,
        "jacobian_evaluations": steps_run,
        "final_loss": history[-1] if history else None,
    }
    return (func_params, stats) if return_stats else func_params

def lm_train_inverse(train_dic, LM_iter=10000, mu_update=3, div_factor=2, mul_factor=3, continue_train=False, model_path=None):
    """
    Levenberg-Marquardt optimization for training two parameter sets simultaneously.

    train_dic: {
        "model_u": {name: (func, vm_args, func_args), ...},
        "model_beta": {name: (func, vm_args, func_args), ...},
        "joint": {name: (func, vm_args, func_args), ...}
    }

    mu_update : update \mu every mu_update iterations
    div_factor : \mu <- \mu/div_factor when loss decreases
    mul_factor : \mu <- mul_factor*\mu when loss increases
    """

    # Extract parameters for both models
    func_params_u = next(iter(train_dic["model_u"].values()))[2][0] if "model_u" in train_dic else \
                   next(iter(train_dic["joint"].values()))[2][0]
    func_params_beta = next(iter(train_dic["model_beta"].values()))[2][0] if "model_beta" in train_dic else \
                      next(iter(train_dic["joint"].values()))[2][1]

    x0_u, signature_u = params_unpack(func_params_u)
    x0_beta, signature_beta = params_unpack(func_params_beta)

    # Load checkpoint if continuing training
    if continue_train:
        assert os.path.exists(model_path), f"model_path {model_path} doesn't exist!"
        checkpoint = torch.load(model_path)
        mu = checkpoint["mu"]
        loss_sum_old = checkpoint["loss_sum_old"]
        itera = checkpoint["itera"]
        history = checkpoint["history"]
        func_params_u = checkpoint["func_params_u"]
        func_params_beta = checkpoint["func_params_beta"]
        x0_u, signature_u = params_unpack(func_params_u)
        x0_beta, signature_beta = params_unpack(func_params_beta)
    else:
        mu = torch.tensor(10**5).to(x0_u)
        loss_sum_old = 10**10
        itera = 0
        history = []

    # Initialize algorithm parameters
    mul_factor = torch.tensor(mul_factor).to(x0_u)
    div_factor = torch.tensor(div_factor).to(x0_u)

    # Combine parameters for joint optimization
    x0 = torch.cat([x0_u, x0_beta])
    I = torch.eye(x0.shape[0]).to(x0)
    x = x0

    time1 = time.time()
    count = 0

    # Define helper function to split parameters
    def split_params(x):
        u_params = x[:len(x0_u)]
        beta_params = x[len(x0_u):]
        return u_params, beta_params

    # Main optimization loop
    for step in range(LM_iter + 1):
        # Split parameters into u and beta components
        x_u, x_beta = split_params(x)
        func_params_u_ = params_pack(x_u, signature_u)
        func_params_beta_ = params_pack(x_beta, signature_beta)

        # Initialize vectors and matrices for residuals and Jacobians
        L_vec = []
        J_mat = []

        # Process model_u loss functions
        if "model_u" in train_dic:
            for loss_func, vm_args, func_args in train_dic["model_u"].values():
                L_item = vmap(loss_func, vm_args)(func_params_u_, *func_args[1:])
                L_item /= np.sqrt(len(L_item))
                L_vec.append(L_item)

                # Get gradients for model_u parameters only
                per_sample_grads = vmap(jacrev(loss_func), vm_args)(func_params_u_, *func_args[1:])
                cnt = 0
                for g in per_sample_grads.values():
                    g = g.detach()
                    J_item = g.view(len(g), -1) if cnt == 0 else torch.hstack([J_item, g.view(len(g), -1)])
                    cnt = 1

                # Pad for beta parameters (zeros)
                J_item = torch.hstack([J_item, torch.zeros((J_item.shape[0], len(x_beta)), device=J_item.device)])
                J_mat.append(J_item)

        # Process model_beta loss functions
        if "model_beta" in train_dic:
            for loss_func, vm_args, func_args in train_dic["model_beta"].values():
                L_item = vmap(loss_func, vm_args)(func_params_beta_, *func_args[1:])
                L_item /= np.sqrt(len(L_item))
                L_vec.append(L_item)

                # Get gradients for model_beta parameters only
                per_sample_grads = vmap(jacrev(loss_func), vm_args)(func_params_beta_, *func_args[1:])
                cnt = 0
                for g in per_sample_grads.values():
                    g = g.detach()
                    J_item = g.view(len(g), -1) if cnt == 0 else torch.hstack([J_item, g.view(len(g), -1)])
                    cnt = 1

                # Pad for u parameters (zeros at beginning)
                J_item = torch.hstack([torch.zeros((J_item.shape[0], len(x_u)), device=J_item.device), J_item])
                J_mat.append(J_item)

        # Process joint loss functions
        if "joint" in train_dic:
            for loss_func, vm_args, func_args in train_dic["joint"].values():
                # Create a wrapper to compute joint loss with separate parameter sets
                def joint_loss_wrapper(params_u, params_beta, *args):
                    return loss_func(params_u, params_beta, *args)

                L_item = vmap(joint_loss_wrapper, vm_args)(func_params_u_, func_params_beta_, *func_args[2:])
                L_item /= np.sqrt(len(L_item))
                L_vec.append(L_item)

                # Compute Jacobian for both parameter sets
                grad_u = vmap(jacrev(joint_loss_wrapper, argnums=0), vm_args)(func_params_u_, func_params_beta_, *func_args[2:])
                grad_beta = vmap(jacrev(joint_loss_wrapper, argnums=1), vm_args)(func_params_u_, func_params_beta_, *func_args[2:])

                # Process u gradients
                cnt_u = 0
                for g in grad_u.values():
                    g = g.detach()
                    J_item_u = g.view(len(g), -1) if cnt_u == 0 else torch.hstack([J_item_u, g.view(len(g), -1)])
                    cnt_u = 1

                # Process beta gradients
                cnt_beta = 0
                for g in grad_beta.values():
                    g = g.detach()
                    J_item_beta = g.view(len(g), -1) if cnt_beta == 0 else torch.hstack([J_item_beta, g.view(len(g), -1)])
                    cnt_beta = 1

                # Combine gradients for both models
                J_item = torch.hstack([J_item_u, J_item_beta])
                J_mat.append(J_item)

        # Combine all residuals and Jacobians
        J_mat = torch.cat(J_mat) if J_mat else torch.zeros((0, len(x)))
        L_vec = torch.cat(L_vec) if L_vec else torch.zeros(0)

        with torch.no_grad():
            # Compute loss
            loss = torch.sum(L_vec**2) if len(L_vec) > 0 else torch.tensor(0.0)
            history.append(loss.item())

            # Update parameters
            J_product = J_mat.t() @ J_mat
            rhs = -J_mat.t() @ L_vec

            # Solve LM system
            L, info = torch.linalg.cholesky_ex(J_product + mu*I)
            if info == 0:
                dp = torch.linalg.solve_triangular(L.t(), torch.linalg.solve_triangular(L, rhs, upper=False), upper=True).squeeze()
            else:  # reject
                mu = mu * mul_factor
                itera += 1
                continue

            x_new = x + dp

            # Evaluate new loss
            x_u_new, x_beta_new = split_params(x_new)
            func_params_u_new = params_pack(x_u_new, signature_u)
            func_params_beta_new = params_pack(x_beta_new, signature_beta)

            loss_new = 0

            # Recompute loss with new parameters
            if "model_u" in train_dic:
                for loss_func, vm_args, func_args in train_dic["model_u"].values():
                    L_item = vmap(loss_func, vm_args)(func_params_u_new, *func_args[1:])
                    L_item /= np.sqrt(len(L_item))
                    loss_new = loss_new + torch.sum(L_item**2)

            if "model_beta" in train_dic:
                for loss_func, vm_args, func_args in train_dic["model_beta"].values():
                    L_item = vmap(loss_func, vm_args)(func_params_beta_new, *func_args[1:])
                    L_item /= np.sqrt(len(L_item))
                    loss_new = loss_new + torch.sum(L_item**2)

            if "joint" in train_dic:
                for loss_func, vm_args, func_args in train_dic["joint"].values():
                    def joint_loss_wrapper(params_u, params_beta, *args):
                        return loss_func(params_u, params_beta, *args)

                    L_item = vmap(joint_loss_wrapper, vm_args)(func_params_u_new, func_params_beta_new, *func_args[2:])
                    L_item /= np.sqrt(len(L_item))
                    loss_new = loss_new + torch.sum(L_item**2)

            # Accept/reject step
            if loss_new > loss:  # reject
                mu = mu * mul_factor
                itera += 1
                continue
            else:
                x = x_new

        # Update mu parameter
        itera += 1
        if step % mu_update == 0:
            if loss < loss_sum_old:
                mu = max(mu/div_factor, 10**(-15))
                count = 0
            else:
                mu = min(mul_factor*mu, 10**(3))
                if mu == 10**(3):
                    count += 1
                if count >= 5:
                    print(f"LM algorithm bloomed up, Iter {itera}, Loss_Res: {loss.item():.5e}, mu: {mu:.5e}")
                    break
            loss_sum_old = loss

        # Logging and checkpointing
        if step % 10 == 0:
            print(f'Iter {itera}, Loss_Res: {loss.item():.5e}, mu: {mu:.5e}')

        if step % 1000 == 0 and model_path:
            x_u, x_beta = split_params(x)
            func_params_u = params_pack(x_u, signature_u)
            func_params_beta = params_pack(x_beta, signature_beta)
            _ensure_parent_dir(model_path)
            torch.save({
                "mu": mu,
                "loss_sum_old": loss_sum_old,
                "itera": itera,
                "func_params_u": func_params_u,
                "func_params_beta": func_params_beta,
                "history": history
            }, model_path)

        if step == LM_iter or loss.item() < 10**(-15):
            break

    # Save final results
    time2 = time.time()
    print(f"Using time {time2-time1:.3f}")
    x_u, x_beta = split_params(x)
    func_params_u = params_pack(x_u, signature_u)
    func_params_beta = params_pack(x_beta, signature_beta)

    if model_path:
        _ensure_parent_dir(model_path)
        torch.save({
            "mu": mu,
            "loss_sum_old": loss_sum_old,
            "itera": itera,
            "func_params_u": func_params_u,
            "func_params_beta": func_params_beta,
            "history": history
        }, model_path)
        print(f"Trained results saved at {model_path}. Using time {time2-time1:.2f}s")

    return func_params_u, func_params_beta


# 参数train_dic使用说明：是一个字典，每一项如果要使用compute_loss_fun1(params,*args_fun), 和vmap(compute_loss_fun1, (None,0,0))(params,*args_fun)
#  则 train_dic["compute_loss_fun1"]=(compute_loss_fun1,(None,0,0),(params,*args_fun))

def gdlm_train(train_dic, LM_iter=2000, imethod=2,iaccel=1 ,ibold=2,
               continue_train=False, model_path=None, otherkw=None,
               return_stats=False):
    func_params=next(iter(train_dic.values()))[2][0]
    x0,signature=params_unpack(func_params)

    def func(x):
        func_params_=params_pack(x,signature)
        L_vec=[]
        for loss_func,vm_args,func_args in train_dic.values():
            L_item=vmap(loss_func,vm_args)(func_params_,*func_args[1:])
            if isinstance(L_item, torch.Tensor):  # return single residual
                L_item/=np.sqrt(len(L_item))
                L_vec.append(L_item)
            else:
                for item in L_item:           # return multiple residuals
                    item/=np.sqrt(len(item))
                    L_vec.append(item)
        L_vec=torch.cat(L_vec)
        return L_vec.squeeze(-1).detach()
    def jacb(x):
        func_params_=params_pack(x,signature)
        J_mat=[]
        for loss_func,vm_args,func_args in train_dic.values():
            per_sample_grads = vmap(jacrev(loss_func), vm_args)(func_params_,*func_args[1:])
            cnt = 0
            for g in per_sample_grads.values():
                g = g.detach()
                J_item = g.view(len(g), -1) if cnt == 0 else torch.hstack([J_item, g.view(len(g), -1)])
                cnt = 1
            J_item/=np.sqrt(len(J_item))
            J_mat.append(J_item)

        J_mat=torch.cat(J_mat)
        return J_mat  # already detached

    if continue_train and os.path.exists(model_path):
        checkpoint=torch.load(model_path)
        state=checkpoint["state"]
    else:
        state=None
    time1=time.time()
    if otherkw is None:
        otherkw = {}
    start_step = int(state.get("istep", -1)) + 1 if isinstance(state, dict) else 0
    x, state=geodesiclm(func,x0,jacb,state=state,maxiter=LM_iter,imethod=imethod,iaccel=iaccel,ibold=ibold,frtol=1e-5,print_level=2, avmax=0.1,**otherkw)
    func_params=params_pack(x,signature)
    time2=time.time()

    usetime = time2-time1
    print(f'运行时间:{usetime:.4f}')
    if model_path:
        _ensure_parent_dir(model_path)
        torch.save({ "func_params": func_params, "state":state}, model_path)
        print(f"Trained results saved at {model_path}. Using time {time2-time1:.2f}s")
    final_step = int(state.get("istep", start_step - 1))
    stats = {
        "actual_iterations": max(0, final_step - start_step + 1),
        "cumulative_iterations": final_step + 1,
        "function_evaluations": state.get("nfev"),
        "jacobian_evaluations": state.get("njev"),
        "final_loss": float(state["Cbest"].detach().cpu()) if torch.is_tensor(state.get("Cbest")) else state.get("Cbest"),
    }
    return (func_params, usetime, stats) if return_stats else (func_params, usetime)


def adam_train(gen_train_dic, model, epoch=1000, learning_rate=5.e-3,
               continue_training=False, model_path=None, return_stats=False):
    if continue_training and os.path.exists(model_path):
        checkpoint=torch.load(model_path)
        opt=torch.optim.AdamW([{"params":model.parameters(),'initial_lr': learning_rate}])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        history = checkpoint['history']
        scheduler = torch.optim.lr_scheduler.StepLR(opt,last_epoch=start_epoch-1,step_size=50,gamma=0.7)
    else:
        start_epoch = 0
        opt=torch.optim.AdamW([{"params":model.parameters()}],lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(opt,step_size=50,gamma=0.7)
        history = []

    initial_history_length = len(history)
    time1=time.time()
    if callable(gen_train_dic):
        generator=gen_train_dic()
    else:
        generator=gen_train_dic
    train_dic = next(generator)
    printed_components = False  # 仅在训练第一步打印各部分loss
    for iter in range(start_epoch, start_epoch+epoch):
        # 每个epoch内部进行若干次“小步”更新，这里采用固定50次全批次迭代
        for i in range(50):
            caches_clear()
            train_dic = next(generator)
            # 仅在第一次小步更新时，逐项计算并打印loss明细
            if not printed_components:
                components = {}
                loss = 0.0
                # 支持两种结构：
                # 1) {name: (loss_fn, data)}
                # 2) {group: {name: (loss_fn, data)}}
                for key, val in train_dic.items():
                    if isinstance(val, dict):
                        for subkey, subval in val.items():
                            loss_fn, data = subval
                            li = loss_fn(model, *data)
                            components[f"{key}/{subkey}"] = li.detach().item()
                            loss = loss + li
                    else:
                        loss_fn, data = val
                        li = loss_fn(model, *data)
                        components[f"{key}"] = li.detach().item()
                        loss = loss + li
                # 打印明细
                print("[loss breakdown]")
                for name, val in components.items():
                    print(f"  - {name}: {val:.4e}")
                printed_components = True
            else:
                loss = 0.0
                for loss_fn, data in train_dic.values():
                    loss = loss + loss_fn(model, *data)
            opt.zero_grad()
            # 使用 autograd.grad 手动赋值，规避二次 backward 触发错误
            grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], retain_graph=False)
            for p, g in zip([p for p in model.parameters() if p.requires_grad], grads):
                p.grad = g
            # 可选梯度裁剪（如需）：
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            opt.step()
            history.append(loss.item())
            # 反向传播后清理一次缓存释放计算图引用
            caches_clear()
        print(f"epoch: {iter}, loss: {loss.item():.4e} ")
        scheduler.step()
        clear_cuda_cache()
        if iter%20 ==19:
            if model_path:
                _ensure_parent_dir(model_path)
                torch.save({ "state_dict": model.state_dict(),"epoch":iter, "history": history}, model_path)


    time2=time.time()
    usetime = time2-time1
    if model_path:
        _ensure_parent_dir(model_path)
        torch.save({ "state_dict": model.state_dict(),"epoch":iter, "history":history}, model_path)
        print(f"Trained results saved at {model_path}. Using time {time2-time1:.2f}s")
    stats = {
        "actual_iterations": len(history) - initial_history_length,
        "epochs_completed": epoch,
        "updates_per_epoch": 50,
        "function_evaluations": len(history) - initial_history_length,
        "jacobian_evaluations": None,
        "final_loss": history[-1] if history else None,
    }
    return (usetime, stats) if return_stats else usetime

def gdlm_train_TNN(train_dic, LM_iter=2000, imethod=2, iaccel=1, ibold=2, continue_train=False, model_path=None, otherkw={}):
    """
    Geodesic Levenberg-Marquardt optimization for training two parameter sets simultaneously.

    train_dic: {
        "model_out": {name: (func, vm_args, func_args), ...},
        "joint": {name: (func, vm_args, func_args), ...}
    }
    """
    # Extract parameters for both models
    func_params_out = next(iter(train_dic["model_out"].values()))[2][0] if "model_out" in train_dic else \
                   next(iter(train_dic["joint"].values()))[2][0]
    func_params_in = next(iter(train_dic["model_in"].values()))[2][0] if "model_in" in train_dic else \
                      next(iter(train_dic["joint"].values()))[2][1]

    x0_out, signature_out = params_unpack(func_params_out)
    x0_in, signature_in = params_unpack(func_params_in)

    # Combine parameters for joint optimization
    x0 = torch.cat([x0_out, x0_in])

    # Define helper function to split parameters
    def split_params(x):
        out_params = x[:len(x0_out)]
        in_params = x[len(x0_out):]
        return out_params, in_params

    # Define residual function for geodesiclm
    def func(x):
        # Split parameters and pack them for each model
        x_out, x_in = split_params(x)
        func_params_out_ = params_pack(x_out, signature_out)
        func_params_in_ = params_pack(x_in, signature_in)

        L_vec = []

        # Process model_u loss functions
        if "model_out" in train_dic:
            for loss_func, vm_args, func_args in train_dic["model_out"].values():
                L_item = vmap(loss_func, vm_args)(func_params_out_, *func_args[1:])
                if isinstance(L_item, torch.Tensor):
                    L_item /= np.sqrt(len(L_item))
                    L_vec.append(L_item)
                else:
                    for item in L_item:
                        item /= np.sqrt(len(item))
                        L_vec.append(item)

        # Process model_beta loss functions
        if "model_in" in train_dic:
            for loss_func, vm_args, func_args in train_dic["model_in"].values():
                L_item = vmap(loss_func, vm_args)(func_params_in_, *func_args[1:])
                if isinstance(L_item, torch.Tensor):
                    L_item /= np.sqrt(len(L_item))
                    L_vec.append(L_item)
                else:
                    for item in L_item:
                        item /= np.sqrt(len(item))
                        L_vec.append(item)

        # Process joint loss functions
        if "joint" in train_dic:
            for loss_func, vm_args, func_args in train_dic["joint"].values():
                def joint_loss_wrapper(params_u, params_beta, *args):
                    return loss_func(params_u, params_beta, *args)

                L_item = vmap(joint_loss_wrapper, vm_args)(func_params_out_, func_params_in_, *func_args[2:])
                if isinstance(L_item, torch.Tensor):
                    L_item /= np.sqrt(len(L_item))
                    L_vec.append(L_item)
                else:
                    for item in L_item:
                        item /= np.sqrt(len(item))
                        L_vec.append(item)

        L_vec = torch.cat(L_vec) if L_vec else torch.zeros(0)
        return L_vec.squeeze(-1).detach()

    # Define Jacobian function for geodesiclm
    def jacb(x):
        # Split parameters and pack them for each model
        x_out, x_in = split_params(x)
        func_params_out_ = params_pack(x_out, signature_out)
        func_params_in_ = params_pack(x_in, signature_in)

        J_mat = []

        # Process model_u Jacobians
        if "model_out" in train_dic:
            for loss_func, vm_args, func_args in train_dic["model_out"].values():
                # loss_func signature: (params_out, X_bd, U_bd)
                per_sample_grads = vmap(jacrev(loss_func), vm_args)(func_params_out_, *func_args[1:])
                cnt = 0
                for g in per_sample_grads.values():
                    g = g.detach()
                    J_item = g.view(len(g), -1) if cnt == 0 else torch.hstack([J_item, g.view(len(g), -1)])
                    cnt = 1
                # Normalize and pad zeros for inner params
                J_item /= np.sqrt(len(J_item))
                J_item = torch.hstack([J_item, torch.zeros((J_item.shape[0], len(x_out)), device=J_item.device)])
                J_mat.append(J_item)

        # Process model_beta Jacobians
        if "model_in" in train_dic:
            for loss_func, vm_args, func_args in train_dic["model_in"].values():
                per_sample_grads = vmap(jacrev(loss_func), vm_args)(func_params_in_, *func_args[1:])
                cnt = 0
                for g in per_sample_grads.values():
                    g = g.detach()
                    J_item = g.view(len(g), -1) if cnt == 0 else torch.hstack([J_item, g.view(len(g), -1)])
                    cnt = 1
                J_item /= np.sqrt(len(J_item))
                J_item = torch.hstack([torch.zeros((J_item.shape[0], len(x_in)), device=J_item.device), J_item])
                J_mat.append(J_item)

        # Process joint Jacobians
        if "joint" in train_dic:
            for loss_func, vm_args, func_args in train_dic["joint"].values():
                # joint loss functions signatures vary; we always pass params_out, params_in followed by tensors
                def joint_loss_wrapper(p_out, p_in, *args):
                    return loss_func(p_out, p_in, *args)
                # Compute grads w.r.t. each param dict separately
                grad_out = vmap(jacrev(joint_loss_wrapper, argnums=0), vm_args)(func_params_out_, func_params_in_, *func_args[2:])
                grad_in = vmap(jacrev(joint_loss_wrapper, argnums=1), vm_args)(func_params_out_, func_params_in_, *func_args[2:])
                cnt_u = 0
                for g in grad_out.values():
                    g = g.detach()
                    J_item_u = g.view(len(g), -1) if cnt_u == 0 else torch.hstack([J_item_u, g.view(len(g), -1)])
                    cnt_u = 1
                cnt_beta = 0
                for g in grad_in.values():
                    g = g.detach()
                    J_item_beta = g.view(len(g), -1) if cnt_beta == 0 else torch.hstack([J_item_beta, g.view(len(g), -1)])
                    cnt_beta = 1
                J_item = torch.hstack([J_item_u, J_item_beta])
                J_item /= np.sqrt(len(J_item))
                J_mat.append(J_item)

        total_dim = x0_out.shape[0] + x0_in.shape[0]
        if J_mat:
            J_mat = torch.cat(J_mat)
        else:
            J_mat = torch.zeros((0, total_dim), device=x0_out.device)
        return J_mat

    # Load state if continuing training
    if continue_train and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        state = checkpoint["state"]
    else:
        state = None

    time1 = time.time()
    x, state = geodesiclm(func, x0, jacb, state=state, maxiter=LM_iter, imethod=imethod,
                          iaccel=iaccel, ibold=ibold, frtol=1e-5, print_level=2,
                          avmax=0.1, **otherkw)
    time2 = time.time()
    usetime = time2-time1
    print(f'运行时间:{usetime:.3f}')

    # Split and pack the optimized parameters
    x_u, x_beta = split_params(x)
    func_params_out = params_pack(x_u, signature_out)
    func_params_in = params_pack(x_beta, signature_in)

    # Save results if model_path is provided
    if model_path:
        torch.save({
            "func_params_u": func_params_out,
            "func_params_beta": func_params_in,
            "state": state
        }, model_path)
        print(f"Trained results saved at {model_path}. Using time {time2-time1:.2f}s")

    return func_params_out, func_params_in , usetime

def adam_train_dual(gen_train_dic, model_u, model_beta, epoch=1000, learning_rate=5.e-3, continue_training=False, model_path=None):
    # 初始化和加载模型状态
    if continue_training and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        opt_u = torch.optim.AdamW([{"params": model_u.parameters(), 'initial_lr': learning_rate}])
        opt_beta = torch.optim.AdamW([{"params": model_beta.parameters(), 'initial_lr': learning_rate}])
        start_epoch = checkpoint['epoch']
        model_u.load_state_dict(checkpoint['state_dict_u'])
        model_beta.load_state_dict(checkpoint['state_dict_beta'])
        history = checkpoint['history']
        scheduler_u = torch.optim.lr_scheduler.StepLR(opt_u, last_epoch=start_epoch-1, step_size=50, gamma=0.7)
        scheduler_beta = torch.optim.lr_scheduler.StepLR(opt_beta, last_epoch=start_epoch-1, step_size=50, gamma=0.7)
    else:
        start_epoch = 0
        opt_u = torch.optim.AdamW([{"params": model_u.parameters()}], lr=learning_rate)
        opt_beta = torch.optim.AdamW([{"params": model_beta.parameters()}], lr=learning_rate)
        scheduler_u = torch.optim.lr_scheduler.StepLR(opt_u, step_size=50, gamma=0.8)
        scheduler_beta = torch.optim.lr_scheduler.StepLR(opt_beta, step_size=50, gamma=0.7)
        history = []

    time1 = time.time()
    if callable(gen_train_dic):
        generator = gen_train_dic()
    else:
        generator = gen_train_dic

    train_dic = next(generator)
    for iter in range(start_epoch, start_epoch+epoch):
        for i in range(50):
            train_dic = next(generator)
            loss = 0

            # 处理model_u单独的损失函数
            if "model_u" in train_dic:
                for loss_fn, data in train_dic["model_u"].values():
                    loss = loss + loss_fn(model_u, *data)
            # 处理model_beta单独的损失函数
            if "model_beta" in train_dic:
                for loss_fn, data in train_dic["model_beta"].values():
                    loss = loss + loss_fn(model_beta,*data)
            # 处理需要两个模型的联合损失函数
            if "joint" in train_dic:
                for loss_fn, data in train_dic["joint"].values():
                    loss = loss + loss_fn(model_u, model_beta, *data)


            # 梯度更新
            opt_u.zero_grad()
            opt_beta.zero_grad()
            loss.backward(retain_graph=True)
            opt_u.step()
            opt_beta.step()

            history.append(loss.item())

        if iter % 10 == 0:
            print(f"epoch: {iter}, loss: {loss.item():.5e}")

        scheduler_u.step()
        scheduler_beta.step()
        clear_cuda_cache()

        if iter % 20 == 19:
            if model_path:
                _ensure_parent_dir(model_path)
                torch.save({
                    "state_dict_u": model_u.state_dict(),
                    "state_dict_beta": model_beta.state_dict(),
                    "epoch": iter,
                    "history": history
                }, model_path)

    time2 = time.time()
    print(f"Using time {time2-time1:.3f}")
    if model_path:
        _ensure_parent_dir(model_path)
        torch.save({
            "state_dict_u": model_u.state_dict(),
            "state_dict_beta": model_beta.state_dict(),
            "epoch": iter,
            "history": history
        }, model_path)
        print(f"Trained results saved at {model_path}. Using time {time2-time1:.2f}s")

def lbfgs_train(gen_train_dic, model, max_iter=10000, continue_training=False,
                model_path=None, return_stats=False):
    import itertools
    counter=itertools.count()
    if continue_training and os.path.exists(model_path):
        checkpoint=torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        history = checkpoint['history']
    else:
        history=[]
    # 首次仅打印一次分项loss明细
    printed_components = False
    optLbfgs=torch.optim.LBFGS(model.parameters(),
                           max_iter=max_iter,
                           tolerance_grad=1e-9,
                           tolerance_change=0,
                           line_search_fn="strong_wolfe"
                           )
    if callable(gen_train_dic):
        generator=gen_train_dic()
    else:
        generator=gen_train_dic
    def closure():
        nonlocal printed_components
        optLbfgs.zero_grad()
        train_dic = next(generator)
        loss=0.0
        if not printed_components:
            components = {}
            # 支持两种结构：
            # 1) {name: (loss_fn, data)}
            # 2) {group: {name: (loss_fn, data)}}
            for key, val in train_dic.items():
                if isinstance(val, dict):
                    for subkey, subval in val.items():
                        loss_fn, data = subval
                        li = loss_fn(model, *data)
                        components[f"{key}/{subkey}"] = li.detach().item()
                        loss = loss + li
                else:
                    loss_fn, data = val
                    li = loss_fn(model, *data)
                    components[f"{key}"] = li.detach().item()
                    loss = loss + li
            print("[loss breakdown]")
            for name, val in components.items():
                print(f"  - {name}: {val:.4e}")
            printed_components = True
        else:
            for loss_fn, data in train_dic.values():
                loss = loss + loss_fn(model,*data)
        loss.backward(retain_graph=True)
        history.append(loss.item())
        if next(counter)%100==0:
            clear_cuda_cache()
            print(f"iter{len(history)},loss: {loss.item():.4e}")
            if model_path:
                _ensure_parent_dir(model_path)
                torch.save({ "state_dict": model.state_dict(),"history": history}, model_path)
            if loss.isnan():
                print("Loss is NaN, exiting the program")
                exit()
        return loss
    time1=time.time()
    optLbfgs.step(closure=closure)
    time2=time.time()
    usetime = time2-time1
    print(f'运行时间:{usetime:.4f}')
    if model_path:
        _ensure_parent_dir(model_path)
        torch.save({ "state_dict": model.state_dict(),"history": history}, model_path)
        print(f"Trained results saved at {model_path}. Using time {time2-time1:.2f}s")
    parameter = next(iter(model.parameters()))
    optimizer_state = optLbfgs.state.get(parameter, {})
    stats = {
        "actual_iterations": int(optimizer_state.get("n_iter", 0)),
        "function_evaluations": int(optimizer_state.get("func_evals", len(history))),
        "jacobian_evaluations": None,
        "final_loss": history[-1] if history else None,
    }
    return (usetime, stats) if return_stats else usetime
