import os
import numpy as np
from .data_loader import DataLoader
from .geometry_processor import GeometryProcessor
from .density_field import DensityFieldGenerator
from .sampling_methods import Sampler
from .visualization import save_visualization


class PipelineController:
    """流水线控制器"""
    
    def __init__(self, config):
        self.config = config
        self.loader = DataLoader()
        
    def run_full_pipeline(self):
        """运行完整流水线"""
        # 加载数据
        Y, y_add_list, y_sub_list, y_file = self.loader.load_data(
            self.config['num'], 
            self.config['filename'], 
            self.config['method'], 
            self.config['n_samples'],
            self.config['yita'],
            self.config['nbrs_k'],
            self.config['perplexity']
        )
        
        # 几何处理
        processor = GeometryProcessor(Y, y_add_list, y_sub_list)
        G = processor.compute_geometry_matrix()
                
        # 密度场生成
        density_gen = DensityFieldGenerator(Y, G)
        density_field = density_gen.generate_density_field()
        point_density = density_gen.compute_point_density(density_field)
        
        # 保存密度场
        os.makedirs("F:/sampling_results", exist_ok=True)
        densityfilename = y_file[:-4] + '_point_density.csv'
        np.savetxt(os.path.join("F:/sampling_results", densityfilename), point_density, delimiter=",")

        # 采样
        sampler = Sampler()
        self._run_sampling_comparison(Y, point_density, y_file, sampler)
                    
    def _run_sampling_comparison(self, Y, point_density, y_file, sampler):
        """运行采样比较"""
        for sample_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            # 标准采样方法
            standard_results = {
                'blue_noise': sampler.blue_noise_sampling(Y, sample_rate),
                'random': sampler.random_sampling(Y, sample_rate),
                'farthest': sampler.farthest_point_sampling(Y, sample_rate),
                'svd': sampler.svd_based_sampling(Y, sample_rate),
                'hash': sampler.hashmap_based_sampling(Y, sample_rate)
            }
            
            # 密度感知采样方法
            density_results = {
                'density_blue_noise': sampler.density_blue_noise_sampling(Y, point_density, sample_rate),
                'density_random': sampler.density_random_sampling(Y, point_density, sample_rate),
                'density_farthest': sampler.density_farthest_point_sampling(Y, point_density, sample_rate),
                'density_svd': sampler.density_svd_based_sampling(Y, point_density, sample_rate),
                'density_hash': sampler.density_hashmap_based_sampling(Y, point_density, sample_rate)
            }
            
            # 保存结果
            self._save_results(Y, standard_results, "standard", y_file, sample_rate, point_density)
            self._save_results(Y, density_results, "density_aware", y_file, sample_rate, point_density)
            
            # 可视化
            save_visualization(Y, standard_results, density_results, y_file, sample_rate)
        
    def _save_results(self, data, results, prefix, y_file, sample_rate, point_density):
        """保存结果"""
        # 这里需要从外部传入label，暂时用占位符
        label = np.zeros(data.shape[0])  # 实际使用时需要传入真实标签
               
        for strategy, samples in results.items():
            index_file = np.hstack((data[samples], samples.reshape(-1, 1)))
            index_label_file = np.hstack((index_file, label[samples].reshape(-1, 1)))
            
            index_label_filename = y_file[:-4] + '_' + f"{prefix}_{strategy}_index_label_samples" + '_' + str(sample_rate) + '.csv'
            np.savetxt(os.path.join("F:/sampling_results", index_label_filename), index_label_file, delimiter=",")