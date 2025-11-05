import numpy as np
from scipy.stats import multivariate_normal
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm


class DensityFieldGenerator:
    """密度场生成器"""
    
    def __init__(self, Y, G, grid_size=200, bandwidth_scale=0.5):
        self.Y = Y
        self.G = G
        self.grid_size = grid_size
        self.bw_scale = bandwidth_scale

        # 计算网格范围
        all_x = np.concatenate([Y[:, 0], [g[0] for g in G]])
        all_y = np.concatenate([Y[:, 1], [g[1] for g in G]])
        self.x_min, self.x_max = all_x.min(), all_x.max()
        self.y_min, self.y_max = all_y.min(), all_y.max()

        # 创建网格
        self.x_grid = np.linspace(self.x_min, self.x_max, grid_size)
        self.y_grid = np.linspace(self.y_min, self.y_max, grid_size)
        self.xx, self.yy = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')
        self.grid_points = np.vstack([self.xx.ravel(), self.yy.ravel()]).T

    def _create_covariance(self, major, minor, angle):
        """创建协方差矩阵"""
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]])
        scaling = np.diag([(major*self.bw_scale)**2, 
                          (minor*self.bw_scale)**2])
        return rotation @ scaling @ rotation.T
    
    def generate_density_field(self):
        """生成密度场"""
        density = np.zeros(self.grid_points.shape[0])
        
        for i in tqdm(range(len(self.G)), desc="生成密度场"):
            cx, cy, major, minor, angle = self.G[i]
            mean = np.array([cx, cy])
            cov = self._create_covariance(major, minor, angle)

            try:
                rv = multivariate_normal(mean, cov + 1e-6*np.eye(2))
                density += rv.pdf(self.grid_points)
            except:
                # 处理数值不稳定的情况
                pass

        return density.reshape(self.xx.shape)

    def compute_point_density(self, density_field):
        """计算点密度"""
        interp_fn = RegularGridInterpolator(
            (self.x_grid, self.y_grid), 
            density_field,
            method='linear',
            bounds_error=False,
            fill_value=1e-10
        )
        return interp_fn(self.Y)