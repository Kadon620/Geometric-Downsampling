import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def local_pca_dn(data):
    """
    执行局部PCA分析
    
    Args:
        data: 输入数据矩阵 (n_samples, n_features)
    
    Returns:
        vectors: 特征向量
        values: 特征值
    """
    n, m = data.shape
    
    if n <= m:
        # 当样本数小于等于特征数时
        mean_data = np.mean(data, axis=0)
        X = data - mean_data
        C = np.matmul(np.transpose(X), X) / n
        eigen_values, eigen_vectors = np.linalg.eig(C)
        
        # 排序特征值
        eig_idx = np.argpartition(eigen_values, -m)[-m:]
        eig_idx = eig_idx[np.argsort(-eigen_values[eig_idx])]
        vectors = eigen_vectors[:, eig_idx]
        vectors = np.transpose(vectors)
        values = eigen_values[eig_idx]
    else:
        # 当样本数大于特征数时
        local_pca = PCA(n_components=data.shape[1], copy=True, whiten=True)
        local_pca.fit(data)
        vectors = local_pca.components_
        values = local_pca.explained_variance_
    
    return vectors, values


def KNN(data, k):
    """
    K近邻搜索
    
    Args:
        data: 输入数据
        k: 近邻数量
    
    Returns:
        index: 近邻索引
    """
    nbr_s = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
    distance, index = nbr_s.kneighbors(data)
    return index


class LocalPCAAnalyzer:
    """局部PCA分析器"""
    
    def __init__(self, n_neighbors=20, max_eigen_count=5):
        self.n_neighbors = n_neighbors
        self.max_eigen_count = max_eigen_count
    
    def analyze(self, X):
        """
        对数据进行局部PCA分析
        
        Args:
            X: 输入数据 (n_samples, n_features)
        
        Returns:
            eigen_vectors_list: 特征向量列表
            eigen_weights: 特征权重
        """
        n, dim = X.shape
        
        if self.max_eigen_count > dim:
            print("Warning: the input MAX_EIGEN_COUNT is too large.")
            self.max_eigen_count = dim
        
        # 计算K近邻
        knn = KNN(X, self.n_neighbors)
        
        # 初始化存储
        eigen_vectors_list = []
        eigen_values = np.zeros((n, dim))
        eigen_weights = np.ones((n, dim))
        
        for i in range(self.max_eigen_count):
            eigen_vectors_list.append(np.zeros((n, dim)))
        
        print('开始执行局部PCA分析……')
        
        # 对每个点进行局部PCA
        for i in tqdm(range(n), desc="局部PCA分析"):
            local_data = np.zeros((self.n_neighbors, dim))
            for j in range(self.n_neighbors):
                local_data[j, :] = X[knn[i, j], :]
            
            temp_vectors, eigen_values[i, :] = local_pca_dn(local_data)
            
            for j in range(self.max_eigen_count):
                eigenvectors = eigen_vectors_list[j]
                eigenvectors[i, :] = temp_vectors[j, :]
            
            # 计算权重
            temp_eigen_sum = sum(eigen_values[i, :])
            for j in range(dim):
                eigen_weights[i, j] = eigen_values[i, j] / temp_eigen_sum
        
        print('执行局部PCA分析结束……')
        return eigen_vectors_list, eigen_weights


def normalize(x, low=-1, up=1):
    """
    数据归一化
    
    Args:
        x: 输入数据
        low: 归一化下限
        up: 归一化上限
    
    Returns:
        new_x: 归一化后的数据
    """
    data_shape = x.shape
    n, dim = data_shape
    new_x = np.zeros(data_shape)
    
    min_v = np.zeros((1, dim))
    max_v = np.zeros((1, dim))
    
    for i in range(dim):
        min_v[0, i] = min(x[:, i])
        max_v[0, i] = max(x[:, i])
    
    for i in range(n):
        for j in range(dim):
            if min_v[0, j] == max_v[0, j]:
                new_x[i, j] = 0
                continue
            new_x[i, j] = ((x[i, j] - min_v[0, j]) / (max_v[0, j] - min_v[0, j])) * (up - low) + low
    
    return new_x