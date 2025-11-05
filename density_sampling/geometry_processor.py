import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm


class GeometryProcessor:
    """几何处理器"""
    
    def __init__(self, Y, y_add_list, y_sub_list):
        self.Y = Y
        self.y_add_list = y_add_list
        self.y_sub_list = y_sub_list
        self.n = Y.shape[0]
        self.p = len(y_add_list)
        
    def _compute_perturbation_vectors(self):
        """计算扰动向量"""
        perturbations = []
        for i in range(self.p):
            add_vec = self.y_add_list[i] - self.Y
            sub_vec = self.Y - self.y_sub_list[i]
            perturbations.extend([add_vec, sub_vec])
        return np.array(perturbations)
    
    def compute_geometry_matrix(self):
        """计算几何矩阵"""
        perturbations = self._compute_perturbation_vectors()
        G = np.zeros((self.n, 5))
        
        for i in tqdm(range(self.n), desc="计算椭圆参数"):
            vecs = perturbations[:, i, :]
            cov = np.cov(vecs.T)

            # 特征值分解
            eigvals, eigvecs = eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            # 计算椭圆参数
            major = 3 * np.sqrt(eigvals[0])
            minor = 3 * np.sqrt(eigvals[1])
            angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
            
            G[i] = [self.Y[i, 0], self.Y[i, 1], major, minor, angle]
            
        return G