import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
import numpy as np
import torch

def visualize_data(data=None,xy=None, u=None,velocity = None,xy_dict='coords',u_dict='u', mode='2d', cmap='jet',show=True):
    """
    数据可视化函数
    参数：
        data   : 数据字典（可选）
        xy     : 坐标张量/数组，形状为[N, 2]
        u      : 场值张量/数组，形状为[N]
        region : 区域标记张量/数组，形状为[N]（可选）
        mode   : 可视化模式，可选 '2d', '3d', 'scatter'
        figsize: 图像尺寸，默认(10,6)
    """
    #x, y, u, region = data[xy_dict][:,0], data[xy_dict][:,1], data[u_dict],data[u_region]

    def to_numpy(tensor):
        """自动转换PyTorch张量为numpy数组,其他类型保持原样"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor) # 非张量数据（如numpy数组）直接返回
    if data is not None:
        # 提取并转换数据
        x = to_numpy(data[xy_dict][:, 0])
        y = to_numpy(data[xy_dict][:, 1])
        u = to_numpy(data[u_dict]).flatten()  # 确保一维数组
    else:
        # 张量直接输入模式
        assert xy is not None , "必须提供xy参数"
        xy = to_numpy(xy)
        x = xy[:, 0]
        y = xy[:, 1]
        # 若提供了三维坐标，提取 z 用于 3D 散点
        z = None
        if isinstance(xy, np.ndarray) and xy.ndim == 2 and xy.shape[1] >= 3:
            z = xy[:, 2]
        u = to_numpy(u).flatten()
        velocity = to_numpy(velocity)


    # 模式1：二维等高线图
    if mode == '2d':
        # 绘制结构化数据
        fig = plt.figure(figsize=(8, 6))
        # cont = plt.tricontourf(x, y, u, levels=50, cmap=cmap)
        n_sqrt = int(np.sqrt(len(x)))  # 假设数据是正方形网格
        x_grid = x.reshape(n_sqrt, n_sqrt)
        y_grid = y.reshape(n_sqrt, n_sqrt)
        u_grid = u.reshape(n_sqrt, n_sqrt)

        # 使用完整坐标信息创建pcolormesh
        cont = plt.pcolormesh(x_grid, y_grid, u_grid, cmap=cmap)
        #cont = plt.pcolormesh(u.reshape(n_sqrt, n_sqrt), cmap=cmap)  # 添加网格线

        # 添加颜色条和标注
        plt.colorbar(cont, label='Value', pad=0.05)
        plt.title('2D Visualization')
        plt.xlabel('x')
        plt.ylabel('y')


        # # 固定坐标范围
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)

    elif mode =='2d_error':
         # 绘制结构化数据
        fig = plt.figure(figsize=(8, 6))
        # cont = plt.tricontourf(x, y, u, levels=50, cmap=cmap)
        n_sqrt = int(np.sqrt(len(x)))  # 假设数据是正方形网格
        x_grid = x.reshape(n_sqrt, n_sqrt)
        y_grid = y.reshape(n_sqrt, n_sqrt)
        u_grid = u.reshape(n_sqrt, n_sqrt)

        # 使用完整坐标信息创建pcolormesh
        cont = plt.pcolormesh(x_grid, y_grid, u_grid, cmap=cmap)
        #cont = plt.pcolormesh(u.reshape(n_sqrt, n_sqrt), cmap=cmap)  # 添加网格线

        # 添加颜色条和标注
        cbar = plt.colorbar(cont, label='Value', pad=0.05)
        from matplotlib.ticker import FuncFormatter
        def format_tick(value, pos):
            return f"{value:.1e}"  # 科学计数法格式
        cbar.formatter = FuncFormatter(format_tick)
        cbar.update_ticks()  # 更新颜色条的刻度格式

        plt.title('2D Visualization')
        plt.xlabel('x')
        plt.ylabel('y')

    elif mode == '2d_scatter':
        # 直接使用三角剖分绘制非结构化数据
        fig = plt.figure(figsize=(8, 6))
        cont = plt.tripcolor(x, y, u, shading='flat', cmap=cmap)
        #cont = plt.tricontourf(x, y, u, levels=50, cmap=cmap)

        #添加颜色条和标注
        plt.colorbar(cont, label='Value', pad=0.05)
        plt.title('2D Visualization')
        plt.xlabel('x')
        plt.ylabel('y')

        # 固定坐标范围
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

    # 模式2：三维图
    elif mode == '3d':
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        # 若含有 z 坐标，则绘制 3D 散点图，并用 u 作为颜色
        if 'z' in locals() and z is not None:
            sc = ax.scatter(x, y, z, c=u, s=4, cmap=cmap, alpha=0.85)
            plt.colorbar(sc, ax=ax, label='u Value')
            ax.set_title('3D Scatter: u(x,y,z)')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        else:
            # 仅有 (x,y) 时，退化为表面三角绘制（旧行为）
            surf = ax.plot_trisurf(x, y, u, cmap=cmap, alpha=0.8)
            plt.colorbar(surf, ax=ax, label='u Value')
            ax.set_title('3D Surface Visualization')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')

    # 模式2.1：3D 点云的连通“表面”渲染（使用凸包近似）
    elif mode == '3d_surface':
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        if not ('z' in locals() and z is not None):
            raise ValueError("3d_surface 模式需要提供三维坐标 xy[:,0:3]")
        try:
            from scipy.spatial import ConvexHull
        except Exception as e:
            # 回退为散点图
            sc = ax.scatter(x, y, z, c=u, s=4, cmap=cmap, alpha=0.85)
            plt.colorbar(sc, ax=ax, label='u Value')
            ax.set_title('3D Scatter (fallback, SciPy missing)')
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
            plt.tight_layout()
            return fig

        pts = np.column_stack([x, y, z])
        hull = ConvexHull(pts)

        # 为每个面计算颜色（取该三角形顶点 u 的平均值）
        tris = hull.simplices  # (M, 3)
        face_vals = np.mean(u[tris], axis=1)
        norm = Normalize(vmin=np.min(u), vmax=np.max(u))
        cm = plt.get_cmap(cmap)
        face_colors = cm(norm(face_vals))

        verts = [pts[tri] for tri in tris]
        mesh = Poly3DCollection(verts, facecolors=face_colors, edgecolors='none', alpha=0.95)
        ax.add_collection3d(mesh)

        # 等比例坐标轴
        max_range = (pts.max(axis=0) - pts.min(axis=0)).max()
        mid = pts.mean(axis=0)
        ax.set_xlim(mid[0]-max_range/2, mid[0]+max_range/2)
        ax.set_ylim(mid[1]-max_range/2, mid[1]+max_range/2)
        ax.set_zlim(mid[2]-max_range/2, mid[2]+max_range/2)

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='u Value')
        ax.set_title('3D Surface (Convex Hull) colored by u')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

        # # 固定XY坐标范围
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)

    # 模式3：散点图组合
    elif mode == 'scatter':
        fig = plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(x, y, c=u, s=2, cmap='viridis',alpha=0.8)
        plt.colorbar(scatter, label='u Value')
        plt.title('scatter points Distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-1, 1)  # 子图1坐标限制
        plt.ylim(-1, 1)


    # 模式4：界面点位置二维图
    elif mode == 'interface':
        fig = plt.figure(figsize=(6, 6))
        scatter = plt.scatter(x, y, s=2, alpha=0.6)
        plt.title('interface points')
        plt.xlabel('x')
        plt.ylabel('y')
        # 固定坐标范围
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

    #模式5：绘制移动网格的散点图
    elif  mode == 'moving points':
            fig = plt.figure(figsize=(6, 6))
            plt.figure(figsize=(6, 6))  # 调整画布尺寸为单图
            plt.scatter(x, y, c='red', s=2, alpha=0.8)  # 固定颜色为蓝色，移除cmap参数
            plt.title('Points Positions')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)

    elif mode == 'velocity field':
            fig = plt.figure(figsize=(7, 6))
            plt.quiver(
                 x,y,#点的坐标
                 velocity[:,0],velocity[:,1],#速度场分量
                 color = 'blue',
                 scale=20, #控制箭头长度（值越小箭头越长）
                 scale_units='width',
                 width=0.005,
                 headwidth=3,
                 headlength=4,alpha=0.8
            )
            #叠加散点显示
            #plt.scatter(x, y, c='black', s=2, alpha=0.5,label='Point Positions')  # 固定颜色为蓝色，移除cmap参数
            plt.title('Vector Field of Scattered Points')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.grid(True)

    plt.tight_layout()
    # 可选显示图形
    if show:
        plt.show()

    # 返回figure对象
    return fig

def visualize_3D(data=None,xyz=None, u=None,velocity = None,xy_dict='coords',u_dict='u', mode='3d', cmap='jet',show=True):
    """
    数据可视化3D方程的函数
    参数：
        data   : 数据字典（可选）
        xyz     : 坐标张量/数组，形状为[N, 3]
        u      : 场值张量/数组，形状为[N]
        region : 区域标记张量/数组，形状为[N]（可选）
        mode   : 可视化模式，可选 '2d', '3d', 'scatter'
        figsize: 图像尺寸，默认(10,6)
    """
    #x, y, u, region = data[xy_dict][:,0], data[xy_dict][:,1], data[u_dict],data[u_region]

    def to_numpy(tensor):
        """自动转换PyTorch张量为numpy数组,其他类型保持原样"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor) # 非张量数据（如numpy数组）直接返回
    if data is not None:
        # 提取并转换数据
        x = to_numpy(data[xy_dict][:, 0])
        y = to_numpy(data[xy_dict][:, 1])
        z = to_numpy(data[xy_dict][:, 2])
        u = to_numpy(data[u_dict]).flatten()  # 确保一维数组
    else:
        # 张量直接输入模式
        assert xyz is not None , "必须提供xy参数"
        xyz = to_numpy(xyz)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        u = to_numpy(u).flatten()
        velocity = to_numpy(velocity)


    # 模式1：二维等高线图
    if mode == '2d':
        # 绘制结构化数据
        fig = plt.figure(figsize=(8, 6))
        # cont = plt.tricontourf(x, y, u, levels=50, cmap=cmap)
        n_sqrt = int(np.sqrt(len(x)))  # 假设数据是正方形网格
        x_grid = x.reshape(n_sqrt, n_sqrt)
        y_grid = y.reshape(n_sqrt, n_sqrt)
        u_grid = u.reshape(n_sqrt, n_sqrt)

        # 使用完整坐标信息创建pcolormesh
        cont = plt.pcolormesh(x_grid, y_grid, u_grid, cmap=cmap)
        #cont = plt.pcolormesh(u.reshape(n_sqrt, n_sqrt), cmap=cmap)  # 添加网格线

        # 添加颜色条和标注
        plt.colorbar(cont, label='Value', pad=0.05)
        plt.title('2D Visualization')
        plt.xlabel('x')
        plt.ylabel('y')


        # # 固定坐标范围
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)


    elif mode == '2d_scatter':
        # 直接使用三角剖分绘制非结构化数据
        fig = plt.figure(figsize=(8, 6))
        cont = plt.tripcolor(x, y, u, shading='flat', cmap=cmap)
        #cont = plt.tricontourf(x, y, u, levels=50, cmap=cmap)

        #添加颜色条和标注
        plt.colorbar(cont, label='Value', pad=0.05)
        plt.title('2D Visualization')
        plt.xlabel('x')
        plt.ylabel('y')

        # 固定坐标范围
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

    # 模式2：三维散点/表面图
    elif mode == '3d':
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        # 对于 3D 数据(x,y,z)->u，绘制散点颜色为 u
        sc = ax.scatter(x, y, z, c=u, s=4, cmap=cmap, alpha=0.85)
        plt.colorbar(sc, ax=ax, label='u Value')
        # ax.set_title('3D Scatter: u(x,y,z)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # 使用相同的坐标范围，取三轴中跨度最大的范围
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)
        max_span = max(x_max - x_min, y_max - y_min, z_max - z_min)
        half_span = max_span / 2.0
        x_mid = (x_max + x_min) / 2.0
        y_mid = (y_max + y_min) / 2.0
        z_mid = (z_max + z_min) / 2.0
        ax.set_xlim(x_mid - half_span, x_mid + half_span)
        ax.set_ylim(y_mid - half_span, y_mid + half_span)
        ax.set_zlim(z_mid - half_span, z_mid + half_span)

    # 模式2.1：3D 点云的连通“表面”渲染（使用凸包近似）
    elif mode == '3d_surface':
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        try:
            from scipy.spatial import ConvexHull
        except Exception:
            sc = ax.scatter(x, y, z, c=u, s=4, cmap=cmap, alpha=0.85)
            plt.colorbar(sc, ax=ax, label='u Value')
            ax.set_title('3D Scatter (fallback, SciPy missing)')
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
            plt.tight_layout()
            return fig

        pts = np.column_stack([x, y, z])
        hull = ConvexHull(pts)
        tris = hull.simplices
        face_vals = np.mean(u[tris], axis=1)
        norm = Normalize(vmin=np.min(u), vmax=np.max(u))
        cm = plt.get_cmap(cmap)
        face_colors = cm(norm(face_vals))
        verts = [pts[tri] for tri in tris]
        mesh = Poly3DCollection(verts, facecolors=face_colors, edgecolors='none', alpha=0.95)
        ax.add_collection3d(mesh)
        max_range = (pts.max(axis=0) - pts.min(axis=0)).max()
        mid = pts.mean(axis=0)
        ax.set_xlim(mid[0]-max_range/2, mid[0]+max_range/2)
        ax.set_ylim(mid[1]-max_range/2, mid[1]+max_range/2)
        ax.set_zlim(mid[2]-max_range/2, mid[2]+max_range/2)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='u Value')
        ax.set_title('3D Surface (Convex Hull) colored by u')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

        # # 固定XY坐标范围
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)

    # 模式3：散点图组合
    elif mode == 'scatter':
        fig = plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(x, y, c=u, s=2, cmap='viridis',alpha=0.8)
        plt.colorbar(scatter, label='u Value')
        plt.title('scatter points Distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-1, 1)  # 子图1坐标限制
        plt.ylim(-1, 1)


    plt.tight_layout()
    # 可选显示图形
    if show:
        plt.show()

    # 返回figure对象
    return fig