import torch
import numpy as np
import time
from tqdm import tqdm
from numba import njit, prange


@njit(parallel=True)
def broadcast_to(array, shape):
    """广播数组到指定形状"""
    result = np.empty(shape, dtype=array.dtype)
    for i in prange(shape[0]):
        for j in range(shape[1]):
            result[i, j] = array[j]
    return result


def compute_tsne_hessian_block_kuai(a, Y, Dy, P, Q, block_size, device):
    """计算t-SNE的Hessian矩阵块"""
    n, m = Y.shape
    PQ = P - Q
    E = 1.0 / (1 + Dy**2)
    eye_m = torch.eye(m, device=device)
    
    # 计算对角块
    Y_a = Y[a]
    dY = Y_a - Y  
    
    Wp = PQ[a].view(-1, 1).expand(n, m)
    Wd = E[a].view(-1, 1).expand(n, m)
    wY = Wd * dY
    
    H_sub1 = -2 * (Wp * wY).T @ wY
    H_sub2 = (PQ[a] @ E[a]) * eye_m
    dY_in = -2 * wY + 4 * (Q[a] @ wY)
    H_sub3 = -wY.T @ (Q[a].view(-1, 1).expand(n, m) * dY_in)
    diag_block = (H_sub1 + H_sub2 + H_sub3) * 4
    
    # 计算非对角块
    non_diag_blocks = []
    non_diag_indices = torch.cat([
        torch.arange(0, a, device=device),
        torch.arange(a+1, n, device=device)
    ])
    
    for c in torch.split(non_diag_indices, block_size):
        Y_c = Y[c]
        dYc = Y_c.unsqueeze(1) - Y.unsqueeze(0)
        E_c_sq = E[c]**2
        sub2_1 = torch.bmm(E_c_sq.unsqueeze(1), dYc).squeeze(1)
        sub2_2 = (Q[a]**2).view(1, -1) @ dY
        H_sub2_batch = -4 * torch.einsum('bi,bj->bij', sub2_2, sub2_1)
        
        # 计算其他项
        dY_batch = dY[c]
        outer = torch.einsum('bi,bj->bij', dY_batch, dY_batch)
        PQ_batch = PQ[a, c].view(-1, 1, 1)
        E_batch = E[a, c].view(-1, 1, 1)
        
        H_sub1_batch = -2 * Q[a, c].view(-1,1,1) * E_batch**2 * outer
        H_sub3_batch = 2 * PQ_batch * E_batch**2 * outer - PQ_batch * E_batch * eye_m
        H_batch = (H_sub1_batch + H_sub2_batch + H_sub3_batch) * 4
        
        non_diag_blocks.append((c, H_batch))
    
    return diag_block, non_diag_blocks


def compute_tsne_jacobian_block_kuai(a, c_indices, X, Y, Dy, beta, P0, n, dim, m, device, dtype):
    """计算t-SNE的Jacobian矩阵块"""
    current_E = 1.0 / (1 + Dy[a]**2)  
    Wbeta = 2 * beta.view(-1, 1)      

    # 计算差值
    dY_batch = Y[c_indices] - Y[a]    
    dX_batch = X[c_indices] - X[a]    

    # 计算各项
    term1 = current_E[c_indices].unsqueeze(-1) * dY_batch  
    term2 = P0[a, c_indices].unsqueeze(-1) * Wbeta[c_indices] * dX_batch  

    # 计算贡献
    contrib = torch.einsum('bm,bd->bmd', term1, term2)  

    # 计算行索引
    rows = (c_indices.unsqueeze(-1) * m + torch.arange(m, device=device)).view(-1)  

    # 构建Jacobian块
    J_block = torch.zeros((n * m, dim), dtype=dtype, device=device)
    J_block.scatter_add_(
        0, 
        rows.unsqueeze(-1).expand(-1, dim),  
        contrib.view(-1, dim)                 
    )

    return J_block * (2 / n)


def compute_tsne_optimized_transform_kuai(X, Y, Dy, beta, P0, Q, vectors_list, H_block_size=100, 
                              J_block_c=200, device='cuda', dtype=torch.float32):
    """执行优化的t-SNE变换"""
    begin_time = time.time()
    n, dim = X.shape
    _, m = Y.shape
    eigen_number = len(vectors_list)
    
    # 转换为PyTorch张量
    X_tensor = torch.as_tensor(X, device=device, dtype=dtype)
    Y_tensor = torch.as_tensor(Y, device=device, dtype=dtype)
    Dy_tensor = torch.as_tensor(Dy, device=device, dtype=dtype)
    beta_tensor = torch.as_tensor(beta, device=device, dtype=dtype)
    P0_tensor = torch.as_tensor(P0, device=device, dtype=dtype)
    Q_tensor = torch.as_tensor(Q, device=device, dtype=dtype)
    vectors_tensors = [torch.as_tensor(vec, device=device, dtype=dtype) for vec in vectors_list]
    
    # 初始化结果
    Y_add_results = [torch.zeros((n, m), device=device, dtype=dtype) for _ in range(eigen_number)]
        
    # 对每个点进行处理
    for a in tqdm(range(n), desc="Processing points"):
        # 计算Hessian块
        diag_block, non_diag_blocks = compute_tsne_hessian_block_kuai(
            a, Y_tensor, Dy_tensor, P0_tensor, Q_tensor, H_block_size, device)
        
        # 构建Hessian行
        H_row = torch.zeros(n*m, m, device=device, dtype=dtype)
        
        # 设置对角块
        H_row[a*m:(a+1)*m] = diag_block
        
        # 设置非对角块
        for c, H_batch in non_diag_blocks:
            for i, c_i in enumerate(c):
                H_row[c_i*m:(c_i+1)*m] = H_batch[i]
        
        # 计算伪逆
        H_pseudo_row = torch.linalg.pinv(H_row.unsqueeze(0))  
        
        # 计算Jacobian行
        J_row = torch.zeros(n*m, dim, dtype=dtype, device=device)
        
        for c_start in range(0, n, J_block_c):
            c_end = min(c_start + J_block_c, n)
            c_indices = torch.arange(c_start, c_end, device=device)
            
            J_sub = compute_tsne_jacobian_block_kuai(a, c_indices, X_tensor, Y_tensor, 
                                        Dy_tensor, beta_tensor, P0_tensor,
                                        n, dim, m, device, dtype)
            J_row += J_sub
        
        # 计算投影矩阵
        P_row = -torch.mm(H_pseudo_row.squeeze(0), J_row)  
        
        # 应用变换
        for i in range(eigen_number):
            Y_add_results[i][a] = torch.mv(P_row, vectors_tensors[i][a])
        
        # 清理内存
        del H_row, H_pseudo_row, J_row, P_row
        torch.cuda.empty_cache()
    
    end_time = time.time()
    print(f'计算花费了 {end_time - begin_time:.2f} 秒')
    
    # 转换为numpy数组
    Y_add_list = [result.cpu().numpy() for result in Y_add_results]
    
    # 计算减法列表
    Y_sub_list = [-1 * array for array in Y_add_list]
    
    return Y_add_list, Y_sub_list


class TSNE_Optimized_Vector_Transform_kuai:
    """优化的t-SNE向量变换器"""
    
    def __init__(self, X, Y, Dy, beta, P0, Q):
        self.X = X
        self.Y = Y
        self.Dy = Dy
        self.beta = beta
        self.P0 = P0
        self.Q = Q
        self.n_samples = Y.shape[0]
        self.y_add_list = []
        self.y_sub_list = []
    
    def transform_all_kuai(self, vector_list, weights):
        """执行完整的t-SNE向量变换"""
        eigen_number = len(vector_list)
        start_time = time.time()
        print('开始执行TSNE向量变换……')
        
        # 加权向量
        weighted_vectors = [vector_list[i] * weights[:, i, None] for i in range(eigen_number)]
        
        # 执行变换
        self.y_add_list, self.y_sub_list = compute_tsne_optimized_transform_kuai(
            self.X, self.Y, self.Dy, self.beta, self.P0, self.Q, weighted_vectors)
        
        end_time = time.time()
        print(f'TSNE向量变换结束，共花费 {end_time - start_time:.2f} 秒')
        
        return self.y_add_list, self.y_sub_list