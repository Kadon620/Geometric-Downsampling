import numpy as np
import random
from scipy.spatial.distance import cdist
from .knn_utils import Knn


class Sampler:
    """采样器基类"""
    
    @staticmethod
    def random_sampling(data, sampling_rate):
        """随机采样"""
        n = data.shape[0]
        m = round(n * sampling_rate)
        perm = np.random.permutation(n)
        selected_indexes = perm[:m]
        return selected_indexes

    @staticmethod
    def blue_noise_sampling(data, sampling_rate, failure_tolerance=1000):
        """蓝噪声采样"""
        n, d = data.shape
        m = round(n * sampling_rate)
        selected_indexes = []
    
        k = 2
        X = np.array(data.tolist(), dtype=np.float64)
        neighbor, dist = Knn(X, n, d, k + 1, 1, 1, n)
        radius = np.average(np.sqrt(dist[:, -1]))
    
        count = 0
        while count < m:
            print('已经成功采样', count, '个样本')
            failure_tolerance = min(5000, (n - m) * 0.1)
            perm = np.random.permutation(n)
            fail = 0
            
            for idx in perm:
                if fail > failure_tolerance or count >= m:
                    break
                
                success = True
                for selected_id in selected_indexes:
                    if np.sum((data[idx] - data[selected_id]) ** 2) < radius ** 2:
                        success = False
                        break
                
                if success:
                    count += 1
                    selected_indexes.append(idx)
                else:
                    fail += 1

            radius /= 2
    
        selected_indexes = np.array(selected_indexes)
        return selected_indexes

    @staticmethod
    def farthest_point_sampling(data, sampling_rate):
        """最远点采样"""
        n = data.shape[0]
        m = round(n * sampling_rate)
        selected_indexes = []

        # 随机选择第一个点
        first_choice = random.randint(0, n - 1)
        selected_indexes.append(first_choice)
        count = 1

        # 初始化距离
        dist = cdist(np.array([data[first_choice]]), data, metric="euclidean").reshape(-1)

        while count < m:
            # 选择距离最远的点
            next_choice = dist.argmax()
            selected_indexes.append(next_choice)

            # 更新距离
            new_dist = cdist(np.array([data[next_choice]]), data, metric="euclidean").reshape(-1)
            dist = np.minimum(dist, new_dist)
            count += 1
            
        selected_indexes = np.array(selected_indexes)
        return selected_indexes

    @staticmethod
    def svd_based_sampling(data, sampling_rate):
        """基于SVD的采样"""
        n = data.shape[0]
        m = round(n * sampling_rate)

        # 执行SVD
        u, s, vt = np.linalg.svd(data.T, full_matrices=True)

        # 计算相关性
        corr = np.sum(vt**2, axis=0)

        # 选择相关性最高的点
        selected_indexes = corr.argsort()[-m:]
        return selected_indexes

    @staticmethod
    def hashmap_based_sampling(data, sampling_rate, grid_size=20, threshold=10):
        """基于哈希的采样"""
        n = data.shape[0]
        m = round(n * sampling_rate)

        # 数据归一化
        data = np.clip(data, 0, 1)
        discrete_full_data = np.clip((data * grid_size).astype(np.int64), 0, grid_size - 1)

        # 初始化冻结和计数器
        frozen = np.zeros((grid_size, grid_size)).astype(np.int64)
        counter = np.zeros((grid_size, grid_size)).astype(np.int64)

        # 随机排列
        perm = np.random.permutation(n)
        count = 0
        selected_idx = []
        
        for idx in perm:
            i, j = discrete_full_data[idx]
            counter[i][j] += 1
            
            if not frozen[i][j]:
                selected_idx.append(idx)
                count += 1
                
                # 冻结周围格子
                if i > 0 and j > 0:
                    frozen[i - 1][j - 1] = 1
                if i > 0 and j < grid_size - 1:
                    frozen[i - 1][j + 1] = 1
                if i < grid_size - 1 and j > 0:
                    frozen[i + 1][j - 1] = 1
                if i < grid_size - 1 and j < grid_size - 1:
                    frozen[i + 1][j + 1] = 1
                
                # 如果超过阈值，解冻周围格子
                if counter[i][j] > threshold:
                    if i > 0 and j > 0:
                        frozen[i - 1][j - 1] = 0
                    if i > 0 and j < grid_size - 1:
                        frozen[i - 1][j + 1] = 0
                    if i < grid_size - 1 and j > 0:
                        frozen[i + 1][j - 1] = 0
                    if i < grid_size - 1 and j < grid_size - 1:
                        frozen[i + 1][j + 1] = 0
            
            if count == m:
                break
        
        selected_indexes = np.array(selected_idx)
        return selected_indexes
    
    @staticmethod
    def density_random_sampling(data, density, sampling_rate, replace=False):
        """密度感知随机采样"""
        n = data.shape[0]
        m = round(n * sampling_rate)
        prob = density / (np.sum(density) + 1e-10)
        
        selected = np.array([], dtype=int)
        nonzero_indices = np.where(prob > 1e-10)[0]
        m_nonzero = min(m, len(nonzero_indices))
        
        if m_nonzero > 0:
            selected = np.random.choice(
                nonzero_indices, 
                size=m_nonzero, 
                replace=False,
                p=prob[nonzero_indices]/prob[nonzero_indices].sum()
            )
        
        remaining = m - m_nonzero
        if remaining > 0:
            if replace and len(nonzero_indices) > 0:
                supplement = np.random.choice(
                    nonzero_indices, 
                    size=remaining, 
                    replace=True,
                    p=prob[nonzero_indices]/prob[nonzero_indices].sum()
                )
            else:
                supplement = np.random.choice(
                    n, 
                    size=remaining, 
                    replace=False
                )
            selected = np.concatenate([selected, supplement])
        
        return selected
    
    @staticmethod
    def density_blue_noise_sampling(data, density, sampling_rate, failure_tolerance=1000):
        """密度感知蓝噪声采样"""
        n, d = data.shape
        m = int(np.round(n * sampling_rate))
        selected_indexes = []
        
        k = 2
        X = np.array(data.tolist(), dtype=np.float64)
        neighbor, dist = Knn(X, n, d, k + 1, 1, 1, n)
                
        median_dist = np.median(np.sqrt(dist[:, -1]))
        weights = 1 / (density + 1e-8)
        weights = weights / np.sum(weights)
        base_radius = np.average(np.sqrt(dist[:, -1]), weights=weights)
        base_radius = min(median_dist, base_radius)
        
        density_normalized = np.log1p(density) / (np.log1p(np.max(density)) + 1e-8)
        remaining_points = set(range(n))
        global_fail_counter = 0
        
        while len(selected_indexes) < m and global_fail_counter < 5:
            sorted_indices = sorted(remaining_points, key=lambda x: -density[x])
            fail = 0
            success_count = 0
            
            for idx in sorted_indices:
                if fail > failure_tolerance or len(selected_indexes) >= m:
                    break
                    
                radius_i = base_radius * (0.5 + 0.5 / (density_normalized[idx] + 0.5))
                conflict = False
                
                if len(selected_indexes) > 0:
                    selected_data = data[selected_indexes]
                    min_dist = np.min(np.linalg.norm(selected_data - data[idx], axis=1))
                    if min_dist < radius_i:
                        conflict = True
                
                if not conflict:
                    selected_indexes.append(idx)
                    remaining_points.discard(idx)
                    success_count += 1
                else:
                    fail += 1
            
            if success_count == 0:
                global_fail_counter += 1
                base_radius *= 0.8
            else:
                global_fail_counter = 0
        
        if len(selected_indexes) < m:
            remaining = list(remaining_points)
            supplement = min(m - len(selected_indexes), len(remaining))
            selected_indexes.extend(np.random.choice(remaining, supplement, replace=False))
        
        return np.array(selected_indexes[:m])

    @staticmethod
    def density_farthest_point_sampling(data, density, sampling_rate):
        """密度感知最远点采样"""
        n = data.shape[0]
        m = round(n * sampling_rate)
                
        if np.all(density == 0):
            density = np.ones(n)
        else:
            density = (density - np.min(density)) / (np.max(density) - np.min(density) + 1e-8)
        
        # 从密度最高的点开始
        first_choice = np.argmax(density)
        selected_indexes = [first_choice]
        
        # 初始化距离
        dist = cdist([data[first_choice]], data, 'euclidean').flatten()
        dist_weighted = dist * (1 + density)
        
        for _ in range(1, m):
            next_choice = np.argmax(dist_weighted)
            selected_indexes.append(next_choice)
            new_dist = cdist([data[next_choice]], data, 'euclidean').flatten()
            dist = np.minimum(dist, new_dist)
            dist_weighted = dist * (1 + density)
            
        return np.array(selected_indexes)

    @staticmethod
    def density_svd_based_sampling(data, density, sampling_rate):
        """密度感知SVD采样"""
        n = data.shape[0]
        m = round(n * sampling_rate)
        
        u, s, vt = np.linalg.svd(data.T, full_matrices=True)
        corr = np.sum(vt ** 2, axis=0) * density
        selected_indexes = corr.argsort()[-m:]
        return selected_indexes
    
    @staticmethod
    def density_hashmap_based_sampling(data, density, sampling_rate, grid_size=20, threshold=10):
        """密度感知哈希采样"""
        n = data.shape[0]
        m = round(n * sampling_rate)
        
        data = np.clip(data, 0, 1)
        discrete_data = (data * grid_size).astype(np.int64)
        discrete_data = np.clip(discrete_data, 0, grid_size - 1)
        
        frozen = np.zeros((grid_size, grid_size), dtype=int)
        counter = np.zeros((grid_size, grid_size), dtype=int)
        
        # 按密度排序
        perm = np.argsort(-density)
        count = 0
        selected_idx = []
        
        for idx in perm:
            if count >= m:
                break
            i, j = discrete_data[idx]
            counter[i][j] += 1
            
            if not frozen[i][j]:
                selected_idx.append(idx)
                count += 1
                
                # 冻结周围格子
                if i > 0 and j > 0:
                    frozen[i-1][j-1] = 1
                if i > 0 and j < grid_size - 1:
                    frozen[i-1][j+1] = 1
                if i < grid_size - 1 and j > 0:
                    frozen[i+1][j-1] = 1
                if i < grid_size - 1 and j < grid_size - 1:
                    frozen[i+1][j+1] = 1
                    
                # 如果超过阈值，解冻周围格子
                if counter[i][j] > threshold:
                    if i > 0 and j > 0:
                        frozen[i-1][j-1] = 0
                    if i > 0 and j < grid_size - 1:
                        frozen[i-1][j+1] = 0
                    if i < grid_size - 1 and j > 0:
                        frozen[i+1][j-1] = 0
                    if i < grid_size - 1 and j < grid_size - 1:
                        frozen[i+1][j+1] = 0
             
        selected_idx = np.array(selected_idx)
        return selected_idx