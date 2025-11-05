import os
import torch
import pandas as pd
import numpy as np
from numba import njit, prange
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA  
from sklearn.manifold import MDS
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

import warnings
  
warnings.filterwarnings("ignore")


class DRTrans:
    
    def __init__(self):
        
        
        self.X = None
        
        
        self.Y = None
        
        
        self.label = None
        
        
        self.derivative = None
        
        
        self.point_error = None
        
        
        self.y_add_list = []
        
        
        self.y_sub_list = []
        
        
        self.k = 0
        
        
        self.eigen_number = 2
        
        self.linearity = None

    
    def transform(self, nbrs_k, MAX_EIGEN_COUNT=5, yita=0.3):
        
        
        return self.Y, self.y_add_list, self.y_sub_list


def read_csv_files_in_folder(folder_path):
    dataframes_and_filenames = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path,header = None)
                dataframes_and_filenames.append((data, file))

    return dataframes_and_filenames



def local_pca_dn(data):
    
    (n, m) = data.shape
    
    
    if n <= m:
        
        
        mean_data = np.mean(data, axis=0)
        
        
        X = data - mean_data
        
        
        C = np.matmul(np.transpose(X), X) / n
        
        
        eigen_values, eigen_vectors = np.linalg.eig(C)
        
        
        eig_idx = np.argpartition(eigen_values, -m)[-m:]
        
        
        eig_idx = eig_idx[np.argsort(-eigen_values[eig_idx])]
        
        
        vectors = eigen_vectors[:, eig_idx]
        
        
        vectors = np.transpose(vectors)
        
        
        values = eigen_values[eig_idx]
        
    
    else:
        
        data_shape = data.shape
        local_pca = PCA(n_components=data_shape[1], copy=True, whiten=True)  
        local_pca.fit(data)  
        
        
        vectors = local_pca.components_
        
        
        values = local_pca.explained_variance_
    
    
    return vectors, values


def normalize(x, low=-1, up=1):
    
    
    data_shape = x.shape
    
    
    n = data_shape[0]
    
    
    dim = data_shape[1]
    
    
    new_x = np.zeros(data_shape)
    
    
    min_v = np.zeros((1, dim))
    max_v = np.zeros((1, dim))
    
    
    for i in range(0, dim):
        
        
        min_v[0, i] = min(x[:, i])
        
        
        max_v[0, i] = max(x[:, i])
        
    
    for i in range(0, n):
        for j in range(0, dim):
            
            
            if min_v[0, j] == max_v[0, j]:
                new_x[i, j] = 0
                continue
            
            
            new_x[i, j] = ((x[i, j] - min_v[0, j]) / (max_v[0, j] - min_v[0, j])) * (up - low) + low
    
    
    return new_x


def KNN(data, k):
    
    nbr_s = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
    
    
    distance, index = nbr_s.kneighbors(data)
    
    
    return index



class T_SNE:
    
    def __init__(self, n_component=2, perplexity=100.0):
        
        self.n_component = n_component
        self.perplexity = perplexity
        self.beta = None
        self.kl = []
        self.final_kl = None
        self.final_iter = 0
        self.P = None
        self.P0 = None
        self.Q = None

    def Hbeta(self, D=np.array([]), beta=1.0):
        

        
        P = np.exp(-D.copy() * beta)
        sum_p = sum(P)
        H = np.log(sum_p) + beta * np.sum(D * P) / sum_p
        P = P / sum_p
        return H, P

    def x2p(self, X=np.array([]), tol=1e-12, perplexity=100.0):
         
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        P = np.zeros((n, n))
        beta = np.ones((n, 1)) / np.max(D)  
        logU = np.log(perplexity)

        
        for i in range(n):

            
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
            (H, thisP) = self.Hbeta(Di, beta[i])

            
            Hdiff = H - logU
            tries = 0
            while np.abs(Hdiff) > tol and tries < 1000:  

                
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                
                (H, thisP) = self.Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

        
        
        self.beta = beta
        return P

    def fit_transform(self, X, max_iter=1000, early_exaggerate=True, y_init=None):
        
        (n, d) = X.shape
        no_dims = self.n_component
        initial_momentum = 0.5
        final_momentum = 0.8
        if not early_exaggerate:
            final_momentum = 0.0
        eta = 500
        min_gain = 0.01

        
        Y2 = np.random.randn(n, no_dims)
        if y_init is None:
            Y2 = np.random.randn(n, no_dims)
        else:
            Y2 = y_init
        dY = np.zeros((n, no_dims))
        iY = np.zeros((n, no_dims))
        gains = np.ones((n, no_dims))

        
        P = self.x2p(X, 1e-15, self.perplexity)
        self.P0 = P.copy()
        P = P + np.transpose(P)
        
        P = P / (2*n)
        P = np.maximum(P, 1e-120)
        self.P = P.copy()

        if early_exaggerate:
            P = P * 4.  

        
        Y = Y2
        for iter in range(max_iter):
            
            sum_Y = np.sum(np.square(Y), 1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-120)

            
            PQ = P - Q
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

            
            if iter < 20 and early_exaggerate:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.))
            if not early_exaggerate:
                gains[gains < min_gain] = min_gain
            iY = momentum * iY - (eta / (np.sqrt(0.1*iter+0.1))) * (gains * dY)  
            Y = Y + iY

            
            if iter == 100 and early_exaggerate:
                P = P / 4.

        
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-120)
        self.Q = Q

        return Y


@njit(parallel=True)
def broadcast_to(array, shape):
    result = np.empty(shape, dtype=array.dtype)
    for i in prange(shape[0]):
        for j in range(shape[1]):
            result[i, j] = array[j]
    return result


class MDS_Vector_Transform_Optimized_kuai:
    def __init__(self, X, Y, Dx, Dy, dtype=torch.float32, device='cuda'):
        self.X = X
        self.Y = Y
        self.Dx = Dx
        self.Dy = Dy
        self.n_samples = Y.shape[0]
        self.dtype = dtype
        self.device = device

       
        self.X_tensor = torch.tensor(X, dtype=dtype, device=device)
        self.Y_tensor = torch.tensor(Y, dtype=dtype, device=device)
        self.Dx_tensor = torch.tensor(Dx, dtype=dtype, device=device)
        self.Dy_tensor = torch.tensor(Dy, dtype=dtype, device=device)

    
    def hessian_mds_block_kuai(self, a, m):
    
        n = self.Y_tensor.size(0)
        Y = self.Y_tensor
        Dx = self.Dx_tensor
        Dy = self.Dy_tensor
      
        Dy2 = Dy.clone().fill_diagonal_(1.0)

        
        dY = Y[a] - Y  

        
        w_a = 2 * Dx[a] / (Dy2[a] ** 3)  

        
        sum_terms = 2 * (n - 1) - 2 * torch.sum(Dx[a] / Dy2[a])

        
        weighted_dY = dY * torch.sqrt(w_a).unsqueeze(-1)  
        weighted_sum = torch.einsum('ni,nj->ij', weighted_dY, weighted_dY)
        H_aa = weighted_sum + sum_terms * torch.eye(m, device=self.device, dtype=self.dtype)

        
        mask = torch.arange(n, device=self.device) != a
        c_indices = torch.where(mask)[0]
        dY_c = dY[c_indices]  
        w_a_c = w_a[c_indices]  

        
        left_sub_coeff = -2 + 2 * Dx[a, c_indices] / Dy2[a, c_indices]

        
        outer_products = torch.einsum('ni,nj->nij', dY_c, dY_c)  
        H_sub = left_sub_coeff.unsqueeze(-1).unsqueeze(-1) * torch.eye(m, device=self.device, dtype=self.dtype)
        H_sub -= w_a_c.unsqueeze(-1).unsqueeze(-1) * outer_products

        H_ac = torch.zeros((n, m, m), device=self.device, dtype=self.dtype)
        H_ac[c_indices] = H_sub

        return H_aa, H_ac

    
    def compute_mds_jacobian_block_kuai(self, c, m, d):
        

        n = self.X_tensor.size(0)
        X = self.X_tensor
        Y = self.Y_tensor
        Dx = self.Dx_tensor
        Dy = self.Dy_tensor

        Dx2 = Dx.clone().fill_diagonal_(1.0)
        Dy2 = Dy.clone().fill_diagonal_(1.0)

        
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

        
        Y_a = Y[c]
        X_a = X[c]
        dY_a = Y_a - Y  
        dX_a = X_a - X  
        coeff_a = 1.0 / (Dy2[c] * Dx2[c])  
        weighted_dX = coeff_a.unsqueeze(-1) * dX_a  
        H_sub_a = -2 * torch.einsum('ni,nj->ij', dY_a, weighted_dX)  
        J_block_3d[c] += H_sub_a

        
        J_block = J_block_3d.view(n * m, d)
    

        return J_block

    
    
    def compute_mds_optimized_transform_kuai(self, vector_list, weights, block_size=256):
        
        begin_time = time.time()
        n, d = self.X.shape
        n2, m = self.Y.shape
        assert n == n2, "X和Y行数不一致"

        eigen_number = len(vector_list)

        
        vectors_tensors = [torch.as_tensor(vec, device=self.device, dtype=self.dtype) for vec in vector_list]
        weight_tensors = [torch.as_tensor(weights[:, i], device=self.device, dtype=self.dtype) for i in range(eigen_number)]

        
        Y_add_results = [torch.zeros((n, m), device=self.device, dtype=self.dtype) for _ in range(eigen_number)]

        
        for c in tqdm(range(n), desc="Processing", total=n):
            
            J_block = self.compute_mds_jacobian_block_kuai(c, m, d)
            H_aa, H_ac = self.hessian_mds_block_kuai(c, m)

            
            H_block = torch.zeros((n * m, m), device=self.device, dtype=self.dtype)
            for a in range(n):
                if a == c:
                    H_block[a*m:(a+1)*m, :] = H_aa
                else:
                    H_block[a*m:(a+1)*m, :] = H_ac[a]

            
            H_pseudo_block = torch.linalg.pinv(H_block)

            
            P_block = -torch.matmul(H_pseudo_block, J_block)

            
            for i in range(eigen_number):
                vector = vectors_tensors[i]
                weight = weight_tensors[i]
                weighted_vector = vector[c] * weight[c]
                Y_add_results[i][c] = torch.mv(P_block, weighted_vector)

            
            del H_block, H_pseudo_block, J_block, P_block
            torch.cuda.empty_cache()

        
        Y_add_list = [result.cpu().numpy() for result in Y_add_results]
        Y_sub_list = [-1 * array for array in Y_add_list]

        end_time = time.time()
        print(f'计算花费了 {end_time - begin_time:.2f} 秒')

        return Y_add_list, Y_sub_list

    
    
    
    def transform_all_kuai(self, vector_list, weights):
        
        start_time = time.time()
        print('开始执行MDS向量变换……')

        
        y_add_list, y_sub_list = self.compute_mds_optimized_transform_kuai(
            vector_list, weights)

        end_time = time.time()
        print('MDS向量变换结束，共花费', end_time - start_time, '时间')

        return y_add_list, y_sub_list




def compute_tsne_hessian_block_kuai(a, Y, Dy, P, Q, block_size, device):
    
    
    n, m = Y.shape
    PQ = P - Q
    E = 1.0 / (1 + Dy**2)
    eye_m = torch.eye(m, device=device)
    
    
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
    
    
    current_E = 1.0 / (1 + Dy[a]**2)  
    Wbeta = 2 * beta.view(-1, 1)      

    
    dY_batch = Y[c_indices] - Y[a]    
    dX_batch = X[c_indices] - X[a]    

    
    term1 = current_E[c_indices].unsqueeze(-1) * dY_batch  
    term2 = P0[a, c_indices].unsqueeze(-1) * Wbeta[c_indices] * dX_batch  

    
    contrib = torch.einsum('bm,bd->bmd', term1, term2)  

    
    rows = (c_indices.unsqueeze(-1) * m + torch.arange(m, device=device)).view(-1)  

    
    J_block = torch.zeros((n * m, dim), dtype=dtype, device=device)
    J_block.scatter_add_(
        0, 
        rows.unsqueeze(-1).expand(-1, dim),  
        contrib.view(-1, dim)                 
    )

    return J_block * (2 / n)


def compute_tsne_optimized_transform_kuai(X, Y, Dy, beta, P0, Q, vectors_list, H_block_size=100, 
                              J_block_c=200, device='cuda', dtype=torch.float32):
    
    
    begin_time = time.time()
    n, dim = X.shape
    _, m = Y.shape
    eigen_number = len(vectors_list)
    
    
    X_tensor = torch.as_tensor(X, device=device, dtype=dtype)
    Y_tensor = torch.as_tensor(Y, device=device, dtype=dtype)
    Dy_tensor = torch.as_tensor(Dy, device=device, dtype=dtype)
    beta_tensor = torch.as_tensor(beta, device=device, dtype=dtype)
    P0_tensor = torch.as_tensor(P0, device=device, dtype=dtype)
    Q_tensor = torch.as_tensor(Q, device=device, dtype=dtype)
    vectors_tensors = [torch.as_tensor(vec, device=device, dtype=dtype) for vec in vectors_list]
    
    
    Y_add_results = [torch.zeros((n, m), device=device, dtype=dtype) for _ in range(eigen_number)]
        
    
    for a in tqdm(range(n), desc="Processing points"):
        
        diag_block, non_diag_blocks = compute_tsne_hessian_block_kuai(
            a, Y_tensor, Dy_tensor, P0_tensor, Q_tensor, H_block_size, device)
        
        
        H_row = torch.zeros(n*m, m, device=device, dtype=dtype)
        
        
        H_row[a*m:(a+1)*m] = diag_block
        
        
        for c, H_batch in non_diag_blocks:
            for i, c_i in enumerate(c):
                H_row[c_i*m:(c_i+1)*m] = H_batch[i]
        
        
        H_pseudo_row = torch.linalg.pinv(H_row.unsqueeze(0))  
        
        
        J_row = torch.zeros(n*m, dim, dtype=dtype, device=device)
        
        for c_start in range(0, n, J_block_c):
            c_end = min(c_start + J_block_c, n)
            c_indices = torch.arange(c_start, c_end, device=device)
            
            J_sub = compute_tsne_jacobian_block_kuai(a, c_indices, X_tensor, Y_tensor, 
                                        Dy_tensor, beta_tensor, P0_tensor,
                                        n, dim, m, device, dtype)
            J_row += J_sub
        
        
        P_row = -torch.mm(H_pseudo_row.squeeze(0), J_row)  
        
        for i in range(eigen_number):
            
            Y_add_results[i][a] = torch.mv(P_row, vectors_tensors[i][a])
        
        
        del H_row, H_pseudo_row, J_row, P_row
        torch.cuda.empty_cache()
    
    end_time = time.time()
    print(f'计算花费了 {end_time - begin_time:.2f} 秒')
    
    
    Y_add_list = [result.cpu().numpy() for result in Y_add_results]
    
    
    
    Y_sub_list = [-1 * array for array in Y_add_list]
    
    return Y_add_list, Y_sub_list


class TSNE_Optimized_Vector_Transform_kuai:
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
        eigen_number = len(vector_list)
        start_time = time.time()
        print('开始执行TSNE向量变换……')
        
        
        weighted_vectors = [vector_list[i] * weights[:, i, None] for i in range(eigen_number)]
        
        
        self.y_add_list, self.y_sub_list = compute_tsne_optimized_transform_kuai(
            self.X, self.Y, self.Dy, self.beta, self.P0, self.Q, weighted_vectors)
        
        end_time = time.time()
        print(f'TSNE向量变换结束，共花费 {end_time - start_time:.2f} 秒')
        
        return self.y_add_list, self.y_sub_list



class PCATrans(DRTrans):
    
    def __init__(self, X, label=None):
        
        super().__init__()
        
        
        self.X = X
        
        
        if label is None:
            
            
            self.label = np.ones((X.shape[0], 1))
            
        else:
            
            
            self.label = label
            
        
        self.n_samples = X.shape[0]
        
        
        self.Y = None
        
        self.y_add = None 
        
        self.y_sub = None
        
        
        self.y_add_list = []
        
        self.y_sub_list = []
        
        
        self.derivative = None
        
        self.point_error = None
        
        
        self.k = 0
        
        
        self.eigen_number = 2

    def transform_(self, eigen_weights, eigen_vectors_list, yita):
        
        
        y_add_list = []
        y_sub_list = []
        
        
        (n, dim) = self.X.shape
        
        
        
        for loop_index in range(0, self.eigen_number):
            
            
            eigenvectors = eigen_vectors_list[loop_index]
            
            
            x_add_v = np.zeros((n, dim))
            x_sub_v = np.zeros((n, dim))
            
            
            for i in range(0, n):
                
                
                x_add_v[i, :] = self.X[i, :] + yita * eigen_weights[i, loop_index] * eigenvectors[i, :]
                
                
                x_sub_v[i, :] = self.X[i, :] - yita * eigen_weights[i, loop_index] * eigenvectors[i, :]
            
            
            y_add_v = np.matmul(x_add_v, self.derivative)
            y_sub_v = np.matmul(x_sub_v, self.derivative)

            y_add_list.append(y_add_v)
            y_sub_list.append(y_sub_v)
        
        return y_add_list, y_sub_list
    
    
    def transform(self, nbrs_k, MAX_EIGEN_COUNT=5, yita=0.3):
        
        
        (n, dim) = self.X.shape     
        
        
        if MAX_EIGEN_COUNT > dim:
            print("Warning: the input MAX_EIGEN_COUNT is too large.")
            MAX_EIGEN_COUNT = dim
        
        
        self.k = nbrs_k
        
        
        self.eigen_number = MAX_EIGEN_COUNT
        
        
        knn = KNN(self.X, nbrs_k)
        
        
        eigen_vectors_list = []
        eigen_values = np.zeros((n, dim))
        eigen_weights = np.ones((n, dim))
        
        
        for i in range(0, MAX_EIGEN_COUNT):
            eigen_vectors_list.append(np.zeros((n, dim)))
        
        print('开始执行PCA降维的局部PCA分析……')
        
        
        for i in range(0, n):
            
            
            local_data = np.zeros((nbrs_k, dim))
            for j in range(0, nbrs_k):
                local_data[j, :] = self.X[knn[i, j], :]
            
            
            temp_vectors, eigen_values[i, :] = local_pca_dn(local_data)
            
            
            for j in range(0, MAX_EIGEN_COUNT):
                eigenvectors = eigen_vectors_list[j]
                eigenvectors[i, :] = temp_vectors[j, :]
            
            
            temp_eigen_sum = sum(eigen_values[i, :])
            for j in range(0, dim):
                eigen_weights[i, j] = eigen_values[i, j] / temp_eigen_sum

        print('执行PCA降维的局部PCA分析结束……')

        
        print("Computing dimension reduction...")
        
        
        pca = PCA(n_components=2, copy=True, whiten=True)
        
        
        pca.fit(self.X)
        
        
        P_ = pca.components_
        
        
        P = np.transpose(P_)
        
        
        self.Y = np.matmul(self.X, P)
        
        
        self.derivative = P
        
        self.y_add_list, self.y_sub_list = self.transform_(eigen_weights, eigen_vectors_list, yita)

        return self.Y, self.y_add_list, self.y_sub_list
    
    

class MDSTrans(DRTrans):
    
    def __init__(self, X, label=None, y_init=None, y_precomputed=False):
        
        
        super().__init__()
        
        
        self.X = X
        
        
        if label is None:
            self.label = np.ones((X.shape[0], 1))
        
        else:
            self.label = label
        
        
        self.n_samples = X.shape[0]
        
        
        self.Y = None
        
        self.Y_add = None
        
        self.Y_sub = None
        
        
        self.y_add_list = []
        
        self.y_sub_list = []
        
        
        self.derivative = None
        
        self.point_error = None
        
        
        self.init_y(y_init, y_precomputed)
        
        
        self.k = 0

    
    def init_y(self, y_init, y_precomputed):
        
        
        print("Computing dimension reduction...")
        
        
        if y_precomputed and (not (y_init is None)):
            
            
            mds = MDS(n_components=2, n_init=1, max_iter=1000, eps=0.01)
            Y = mds.fit_transform(self.X, init=y_init)
        
        else:
            
            
            print("使用默认的MDS初始化方式")
            mds = MDS(n_components=2, max_iter=1000, eps=0.01)
            Y = mds.fit_transform(self.X)
            
        print('MDS降维结束，开始向量变换')
        
        
        self.Y = Y

    
    def transform(self, nbrs_k, MAX_EIGEN_COUNT=5, yita=0.3):
        
        
        print("Transform MDS...")
        
        
        (n, dim) = self.X.shape
        
        print(n,dim)
        
        
        if MAX_EIGEN_COUNT > dim:
            print("Warning: the input MAX_EIGEN_COUNT is too large.")
            MAX_EIGEN_COUNT = dim
        
        
        self.k = nbrs_k
        
        
        self.eigen_number = MAX_EIGEN_COUNT
        
        
        knn = KNN(self.X, nbrs_k)
        
        
        eigen_vectors_list = []
        eigen_values = np.zeros((n, dim))
        eigen_weights = np.ones((n, dim))
        
        
        
        print('开始执行MDS的局部PCA分析……')
        
        for i in range(0, MAX_EIGEN_COUNT):
            eigen_vectors_list.append(np.zeros((n, dim)))
        
        
        for i in range(0, n):
            
            
            
            local_data = np.zeros((nbrs_k, dim))
            for j in range(0, nbrs_k):
                local_data[j, :] = self.X[knn[i, j], :]
            
            
            temp_vectors, eigen_values[i, :] = local_pca_dn(local_data)
            
            
            for j in range(0, MAX_EIGEN_COUNT):
                eigenvectors = eigen_vectors_list[j]
                eigenvectors[i, :] = temp_vectors[j, :]
            
            
            temp_eigen_sum = sum(eigen_values[i, :])
            for j in range(0, dim):
                eigen_weights[i, j] = eigen_values[i, j] / temp_eigen_sum
        
        print('执行MDS的局部PCA分析结束……')
        
        
        print("Computing vectors transformation...")
        
        
        self.Dx = euclidean_distances(self.X)
        
        
        self.Dy = euclidean_distances(self.Y)  
        
        
        transformer = MDS_Vector_Transform_Optimized_kuai(self.X, self.Y, self.Dx, self.Dy)
        
        self.y_add_list, self.y_sub_list = transformer.transform_all_kuai(eigen_vectors_list, yita * eigen_weights)
    
        return self.Y, self.y_add_list, self.y_sub_list
   
    

class TSNETrans(DRTrans):
    
    def __init__(self, X, label=None, y_init=None, perplexity=100.0):
        
        
        super().__init__()
        
        
        self.X = X
        
        
        if label is None:
            
            
            self.label = np.ones((X.shape[0], 1))
        
        else:
            
            
            self.label = label
        
        
        self.n_samples = X.shape[0]
        
        
        self.Y = None
        
        self.y_add = None 
        
        self.y_sub = None
        
        
        self.y_add_list = []
        
        self.y_sub_list = []
        
        
        self.P0 = None
        
        
        self.Px = None
        
        
        self.Q = None
        
        
        self.beta = None
        
        
        self.derivative = None
        
        
        self.point_error = None
        
        
        self.init_y(y_init, perplexity=100.0)
        
        
        self.k = 0
        
        
        self.eigen_number = 2
    
    
    
    def init_y(self, Y0, perplexity=100.0):
        
        
        print("Computing dimension reduction...")      
        
        
        t_sne = T_SNE(perplexity=100.0)
        
        
        
        
        if Y0 is None:
            
            
            Y = t_sne.fit_transform(self.X, max_iter=1000)
            
            
            
        
        else:
            
            
            Y = t_sne.fit_transform(self.X, max_iter=1000, early_exaggerate=False, y_init=Y0)
            
            
            
        
        print('t-SNE降维结束，开始向量变换')
        
        
        self.Y = Y
        
        
        
        
        self.beta = t_sne.beta
        
        
        self.P0 = t_sne.P0
        
        
        self.Px = t_sne.P
        
        
        self.Q = t_sne.Q


    def transform(self, nbrs_k, MAX_EIGEN_COUNT=5, yita=0.3):
        
        print("Transform t-SNE...")
        
        
        (n, dim) = self.X.shape
        
        
        if MAX_EIGEN_COUNT > dim:
            print("Warning: the input MAX_EIGEN_COUNT is too large.")
            MAX_EIGEN_COUNT = dim
        
        
        self.k = nbrs_k
        
        
        self.eigen_number = MAX_EIGEN_COUNT
        
        
        knn = KNN(self.X, nbrs_k)
        
        
        eigen_vectors_list = []
        eigen_values = np.zeros((n, dim))
        eigen_weights = np.ones((n, dim))
        
        print('开始执行TSNE的局部PCA分析……')
        
        
        for i in range(0, MAX_EIGEN_COUNT):
            eigen_vectors_list.append(np.zeros((n, dim)))
            
        
        for i in range(0, n):
            
            
            local_data = np.zeros((nbrs_k, dim))
            
            for j in range(0, nbrs_k):
                local_data[j, :] = self.X[knn[i, j], :]
                
            
            temp_vectors, eigen_values[i, :] = local_pca_dn(local_data)
            
            
            for j in range(0, MAX_EIGEN_COUNT):
                eigenvectors = eigen_vectors_list[j]
                eigenvectors[i, :] = temp_vectors[j, :]
                
            
            temp_eigen_sum = sum(eigen_values[i, :])
            
            for j in range(0, dim):
                eigen_weights[i, j] = eigen_values[i, j] / temp_eigen_sum
        
        print('执行TSNE的局部PCA分析结束……')
        
        
        self.Dx = euclidean_distances(self.X)
        
        
        self.Dy = euclidean_distances(self.Y)  
        
        
        
        transformer = TSNE_Optimized_Vector_Transform_kuai(self.X, self.Y, self.Dy, self.beta, self.P0, self.Q)
        
        self.y_add_list, self.y_sub_list = transformer.transform_all_kuai(eigen_vectors_list, yita * eigen_weights)

        return self.Y, self.y_add_list, self.y_sub_list



transform = transforms.Compose([
    
    transforms.Resize((224, 224)),
    
    transforms.ToTensor(),
    
    transforms.Normalize((0.5, ), (0.5, ))
])


train_dataset = datasets.MNIST(root='E:/Data/MNIST', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


test_dataset = datasets.MNIST(root='E:/Data/MNIST', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

X_MNIST = train_dataset.train_data

X_MNIST = np.array(X_MNIST, dtype=np.float64)

y_MNIST = train_dataset.train_labels

y_MNIST = np.array(y_MNIST, dtype=np.float64)

X_MNIST = X_MNIST[:10000,].reshape(-1, 28 * 28)

y_MNIST = y_MNIST[:10000,].reshape(-1,)





np.random.seed(0)


num_samples = 500  
num_features = 64  
overlap_factor = 0.5  
sample_rate = 0.3  


mean1 = np.zeros(num_features)  
cov1 = np.eye(num_features)  
data1 = np.random.multivariate_normal(mean1, cov1, num_samples)


mean2 = overlap_factor * np.ones(num_features)  
cov2 = np.eye(num_features)  
data2 = np.random.multivariate_normal(mean2, cov2, num_samples)


print("Shape of data1:", data1.shape)  
print("Shape of data2:", data2.shape)  


moni_data = np.vstack([data1, data2])
moni_labels = np.hstack([np.zeros(num_samples), np.ones(num_samples)])  
   

    


'''
@misc{mice_protein_expression_342,
  author       = {Higuera, Clara, Gardiner, Katheleen, and Cios, Krzysztof},
  title        = {{Mice Protein Expression}},
  year         = {2015},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C50S3Z}
}
'''


mice_protein_expression = fetch_ucirepo(id=342) 
  

X_mice = mice_protein_expression.data.features 


X_mice_value = X_mice.iloc[:,:-3]


X_mice_value.fillna(X_mice_value.mode().iloc[0], inplace=True)

X_mice_no_nan = X_mice_value

y_mice = mice_protein_expression.data.targets 
  
X_mice_array = np.array(X_mice_no_nan, dtype=np.float64)

y_mice_label = label_encoder.fit_transform(y_mice)

y_mice_array = np.array(y_mice_label, dtype=np.float64)



def run_example():
    
    
    for method in ('MDS','t-SNE','PCA'):
        
        dr_method = method
        
        print("正在进行",dr_method,"方法处理")
        
        
        (n, d) = X.shape
        
        print("样本总共选择", n , '个')

        
        trans = DRTrans()
        print("创建DRTrans基础对象")
        
        
        if dr_method == 'MDS':
            print("实例化MDSTrans对象")
            
            trans = MDSTrans(X, label=label, y_init=None, y_precomputed=False)
            
        elif dr_method == 't-SNE':
            print("实例化TSNETrans对象")
            
            trans = TSNETrans(X, label=label, y_init=None, perplexity=100.0)
            
        elif dr_method == 'PCA':
            print("实例化PCATrans对象")
            
            trans = PCATrans(X, label=label)
        
        else:
            
            print("This method is not supported at this time: ", dr_method)
            return

        
        print("对数据进行降维转换")
        trans.transform(nbrs_k = nbrs_k, MAX_EIGEN_COUNT = MAX_EIGEN_COUNT, yita = yita)
        
        
        np.savetxt("F:/data_new/dr_data_mnist/" + "MNIST_" + str(dr_method) + "_" + str(n) + "_" + str(nbrs_k) + "_Y.csv", trans.Y, fmt='%.18e', delimiter=",")
        
        
        np.savetxt("F:/data_new/dr_data_mnist/" + "MNIST_" + str(dr_method) + "_" + str(n) + "_" + str(nbrs_k) + "_Y_add_list0.csv", trans.y_add_list[0], fmt='%.18e', delimiter=",")
        
        
        np.savetxt("F:/data_new/dr_data_mnist/" + "MNIST_" + str(dr_method) + "_" + str(n) + "_" + str(nbrs_k) + "_Y_sub_list0.csv", trans.y_sub_list[0], fmt='%.18e', delimiter=",")
        
        
        np.savetxt("F:/data_new/dr_data_mnist/" + "MNIST_" + str(dr_method) + "_" + str(n) + "_" + str(nbrs_k) + "_Y_add_list1.csv", trans.y_add_list[1], fmt='%.18e', delimiter=",")
        
        
        np.savetxt("F:/data_new/dr_data_mnist/" + "MNIST_" + str(dr_method) + "_" + str(n) + "_" + str(nbrs_k) + "_Y_sub_list1.csv", trans.y_sub_list[1], fmt='%.18e', delimiter=",")
        
        
        np.savetxt("F:/data_new/dr_data_mnist/" + "MNIST_" + str(dr_method) + "_" + str(n) + "_" + str(nbrs_k) + "_Y_add_list2.csv", trans.y_add_list[2], fmt='%.18e', delimiter=",")
        
        
        np.savetxt("F:/data_new/dr_data_mnist/" + "MNIST_" + str(dr_method) + "_" + str(n) + "_" + str(nbrs_k) + "_Y_sub_list2.csv", trans.y_sub_list[2], fmt='%.18e', delimiter=",")
        
        
        np.savetxt("F:/data_new/dr_data_mnist/" + "MNIST_" + str(dr_method) + "_" + str(n) + "_" + str(nbrs_k) + "_Y_add_list3.csv", trans.y_add_list[3], fmt='%.18e', delimiter=",")
        
        
        np.savetxt("F:/data_new/dr_data_mnist/" + "MNIST_" + str(dr_method) + "_" + str(n) + "_" + str(nbrs_k) + "_Y_sub_list3.csv", trans.y_sub_list[3], fmt='%.18e', delimiter=",")



nbrs_k=20


MAX_EIGEN_COUNT=4


yita=0.5

 
if __name__ == '__main__':
    
    X, label = X_MNIST, y_MNIST
    run_example()
                
            
