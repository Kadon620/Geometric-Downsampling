import torch
import numpy as np
import time
from tqdm import tqdm


class MDS_Vector_Transform_Optimized_kuai:
    """优化的MDS向量变换器"""
    
    def __init__(self, X, Y, Dx, Dy, dtype=torch.float32, device='cuda'):
        self.X = X
        self.Y = Y
        self.Dx = Dx
        self.Dy = Dy
        self.n_samples = Y.shape[0]
        self.dtype = dtype
        self.device = device

        # 转换为PyTorch张量
        self.X_tensor = torch.tensor(X, dtype=dtype, device=device)
        self.Y_tensor = torch.tensor(Y, dtype=dtype, device=device)
        self.Dx_tensor = torch.tensor(Dx, dtype=dtype, device=device)
        self.Dy_tensor = torch.tensor(Dy, dtype=dtype, device=device)

    def hessian_mds_block_kuai(self, a, m):
        """计算MDS的Hessian矩阵块"""
        n = self.Y_tensor.size(0)
        Y = self.Y_tensor
        Dx = self.Dx_tensor
        Dy = self.Dy_tensor
      
        Dy2 = Dy.clone().fill_diagonal_(1.0)

        # 计算差值
        dY = Y[a] - Y  

        # 计算权重
        w_a = 2 * Dx[a] / (Dy2[a] ** 3)  

        # 计算求和项
        sum_terms = 2 * (n - 1) - 2 * torch.sum(Dx[a] / Dy2[a])

        # 计算加权的dY
        weighted_dY = dY * torch.sqrt(w_a).unsqueeze(-1)  
        weighted_sum = torch.einsum('ni,nj->ij', weighted_dY, weighted_dY)
        H_aa = weighted_sum + sum_terms * torch.eye(m, device=self.device, dtype=self.dtype)

        # 计算非对角块
        mask = torch.arange(n, device=self.device) != a
        c_indices = torch.where(mask)[0]
        dY_c = dY[c_indices]  
        w_a_c = w_a[c_indices]  

        # 计算左子系数
        left_sub_coeff = -2 + 2 * Dx[a, c_indices] / Dy2[a, c_indices]

        # 计算外积
        outer_products = torch.einsum('ni,nj->nij', dY_c, dY_c)  
        H_sub = left_sub_coeff.unsqueeze(-1).unsqueeze(-1) * torch.eye(m, device=self.device, dtype=self.dtype)
        H_sub -= w_a_c.unsqueeze(-1).unsqueeze(-1) * outer_products

        H_ac = torch.zeros((n, m, m), device=self.device, dtype=self.dtype)
        H_ac[c_indices] = H_sub

        return H_aa, H_ac

    def compute_mds_jacobian_block_kuai(self, c, m, d):
        """计算MDS的Jacobian矩阵块"""
        n = self.X_tensor.size(0)
        X = self.X_tensor
        Y = self.Y_tensor
        Dx = self.Dx_tensor
        Dy = self.Dy_tensor

        Dx2 = Dx.clone().fill_diagonal_(1.0)
        Dy2 = Dy.clone().fill_diagonal_(1.0)

        # 计算非对角贡献
        mask = torch.arange(n, device=self.device) != c
        a_indices = torch.where(mask)[0]
        
        if len(a_indices) > 0:
            Y_c = Y[c]
            X_c = X[c]
            dY_ac = Y[a_indices] - Y_c  
            dX_ac = X[a_indices] - X_c  
            coeff = 2.0 / (Dy2[a_indices, c] * Dx2[a_indices, c])  
            contrib = coeff.unsqueeze(-1).unsqueeze(-1) * torch.einsum('ki,kj->kij', dY_ac, dX_ac)  
            J_block_3d = torch.zeros((n, m, d), device=self.device, dtype=self.dtype)
            J_block_3d[a_indices] += contrib

        # 计算对角贡献
        Y_a = Y[c]
        X_a = X[c]
        dY_a = Y_a - Y  
        dX_a = X_a - X  
        coeff_a = 1.0 / (Dy2[c] * Dx2[c])  
        weighted_dX = coeff_a.unsqueeze(-1) * dX_a  
        H_sub_a = -2 * torch.einsum('ni,nj->ij', dY_a, weighted_dX)  
        J_block_3d[c] += H_sub_a

        # 重塑Jacobian块
        J_block = J_block_3d.view(n * m, d)
    
        return J_block

    def compute_mds_optimized_transform_kuai(self, vector_list, weights, block_size=256):
        """执行优化的MDS变换"""
        begin_time = time.time()
        n, d = self.X.shape
        n2, m = self.Y.shape
        assert n == n2, "X和Y行数不一致"

        eigen_number = len(vector_list)

        # 转换为PyTorch张量
        vectors_tensors = [torch.as_tensor(vec, device=self.device, dtype=self.dtype) for vec in vector_list]
        weight_tensors = [torch.as_tensor(weights[:, i], device=self.device, dtype=self.dtype) for i in range(eigen_number)]

        # 初始化结果
        Y_add_results = [torch.zeros((n, m), device=self.device, dtype=self.dtype) for _ in range(eigen_number)]

        # 对每个点进行处理
        for c in tqdm(range(n), desc="Processing", total=n):
            # 计算Jacobian和Hessian块
            J_block = self.compute_mds_jacobian_block_kuai(c, m, d)
            H_aa, H_ac = self.hessian_mds_block_kuai(c, m)

            # 构建Hessian块
            H_block = torch.zeros((n * m, m), device=self.device, dtype=self.dtype)
            for a in range(n):
                if a == c:
                    H_block[a*m:(a+1)*m, :] = H_aa
                else:
                    H_block[a*m:(a+1)*m, :] = H_ac[a]

            # 计算伪逆
            H_pseudo_block = torch.linalg.pinv(H_block)

            # 计算投影矩阵
            P_block = -torch.matmul(H_pseudo_block, J_block)

            # 应用变换
            for i in range(eigen_number):
                vector = vectors_tensors[i]
                weight = weight_tensors[i]
                weighted_vector = vector[c] * weight[c]
                Y_add_results[i][c] = torch.mv(P_block, weighted_vector)

            # 清理内存
            del H_block, H_pseudo_block, J_block, P_block
            torch.cuda.empty_cache()

        # 转换为numpy数组
        Y_add_list = [result.cpu().numpy() for result in Y_add_results]
        Y_sub_list = [-1 * array for array in Y_add_list]

        end_time = time.time()
        print(f'计算花费了 {end_time - begin_time:.2f} 秒')

        return Y_add_list, Y_sub_list

    def transform_all_kuai(self, vector_list, weights):
        """执行完整的MDS向量变换"""
        start_time = time.time()
        print('开始执行MDS向量变换……')

        # 执行变换
        y_add_list, y_sub_list = self.compute_mds_optimized_transform_kuai(
            vector_list, weights)

        end_time = time.time()
        print('MDS向量变换结束，共花费', end_time - start_time, '时间')

        return y_add_list, y_sub_list