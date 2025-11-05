import numpy as np
import pandas as pd
import os
import sys
import cffi
import time
import threading
from scipy.spatial.distance import cdist
import random
from scipy.linalg import eigh
from scipy.stats import multivariate_normal
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def read_csv_files_in_folder(folder_path):
    dataframes_and_filenames = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path,header = None)
                dataframes_and_filenames.append((data, file))

    return dataframes_and_filenames 


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


class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        return self._target(*self._args)

def Knn(X, N, D, n_neighbors, forest_size, subdivide_variance_size, leaf_number):

    ffi = cffi.FFI()
    ffi.cdef()

    try:

        t1 = time.time()

        dllPath = os.path.join('F:/knnDll.dll')

        C = ffi.dlopen(dllPath)

        cffi_X1 = ffi.cast('double*', X.ctypes.data)

        neighbors_nn = np.zeros((N, n_neighbors), dtype=np.int32)
        distances_nn = np.zeros((N, n_neighbors), dtype=np.float64)

        cffi_neighbors_nn = ffi.cast('int*', neighbors_nn.ctypes.data)
        cffi_distances_nn = ffi.cast('double*', distances_nn.ctypes.data)

        t = FuncThread(C.knn, cffi_X1, N, D, n_neighbors, cffi_neighbors_nn, cffi_distances_nn, 
                       forest_size, subdivide_variance_size, leaf_number)

        t.daemon = True

        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()

        print("knn runtime = %f" % (time.time() - t1))

        return neighbors_nn, distances_nn

    except Exception as ex:
        print(ex)

    return [[], []]


class DataLoader:

    def __init__(self, base_path="F:/dr_data_mnist_kuai"):
        self.base_path = base_path
        
    def _construct_filename(self, num, filename, method, n_samples, yita, nbrs_k, perplexity, file_type, list_index=None):
        
        parts = [
            f"{num}_{filename}_{method}_{n_samples}_{yita}_{nbrs_k}_{perplexity}",
            "Y" if file_type == "Y" else 
            f"Y_{'add' if file_type.startswith('add') else 'sub'}_list{list_index}"
        ]
        return f"{'_'.join(parts)}.csv"

    def load_data(self, num, filename, method, n_samples, yita, nbrs_k, perplexity):
        

        y_file = self._construct_filename(num, filename, method, n_samples, yita, nbrs_k, perplexity, "Y")
        Y = pd.read_csv(os.path.join(self.base_path, y_file), header=None).values

        y_add_list = []
        y_sub_list = []
        for i in range(4):
            add_file = self._construct_filename(num, filename, method, n_samples, yita, nbrs_k, perplexity, "add", i)
            sub_file = self._construct_filename(num, filename, method, n_samples, yita, nbrs_k, perplexity, "sub", i)
            y_add_list.append(pd.read_csv(os.path.join(self.base_path, add_file), header=None).values)
            y_sub_list.append(pd.read_csv(os.path.join(self.base_path, sub_file), header=None).values)
            
        return Y, y_add_list, y_sub_list, y_file


class GeometryProcessor:
    def __init__(self, Y, y_add_list, y_sub_list):
        self.Y = Y
        self.y_add_list = y_add_list
        self.y_sub_list = y_sub_list
        self.n = Y.shape[0]
        self.p = len(y_add_list)
        
    def _compute_perturbation_vectors(self):
        
        perturbations = []

        for i in range(self.p):
            add_vec = self.y_add_list[i] - self.Y
            sub_vec = self.Y - self.y_sub_list[i]
            perturbations.extend([add_vec, sub_vec])
        return np.array(perturbations)
    
    def compute_geometry_matrix(self):
        
        perturbations = self._compute_perturbation_vectors()
        G = np.zeros((self.n, 5))
        
        for i in tqdm(range(self.n), desc="计算椭圆参数"):

            vecs = perturbations[:, i, :]

            cov = np.cov(vecs.T)

            eigvals, eigvecs = eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            major = 3 * np.sqrt(eigvals[0])
            minor = 3 * np.sqrt(eigvals[1])
            angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
            
            G[i] = [self.Y[i, 0], self.Y[i, 1], major, minor, angle]
            
        return G


class DensityFieldGenerator:
    def __init__(self, Y, G, grid_size=200, bandwidth_scale=0.5):
        self.Y = Y
        self.G = G
        self.grid_size = grid_size
        self.bw_scale = bandwidth_scale

        all_x = np.concatenate([Y[:, 0], [g[0] for g in G]])
        all_y = np.concatenate([Y[:, 1], [g[1] for g in G]])
        self.x_min, self.x_max = all_x.min(), all_x.max()
        self.y_min, self.y_max = all_y.min(), all_y.max()

        self.x_grid = np.linspace(self.x_min, self.x_max, grid_size)
        self.y_grid = np.linspace(self.y_min, self.y_max, grid_size)
        self.xx, self.yy = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')
        self.grid_points = np.vstack([self.xx.ravel(), self.yy.ravel()]).T


    def _create_covariance(self, major, minor, angle):
        
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]])

        scaling = np.diag([(major*self.bw_scale)**2, 
                          (minor*self.bw_scale)**2])
        return rotation @ scaling @ rotation.T
    
    def generate_density_field(self):
        
        density = np.zeros(self.grid_points.shape[0])
        
        for i in tqdm(range(len(self.G)), desc="生成密度场"):
            cx, cy, major, minor, angle = self.G[i]
            mean = np.array([cx, cy])
            cov = self._create_covariance(major, minor, angle)

            try:
                rv = multivariate_normal(mean, cov + 1e-6*np.eye(2))
                density += rv.pdf(self.grid_points)
            except:

                pass



        return density.reshape(self.xx.shape)

    def compute_point_density(self, density_field):
        interp_fn = RegularGridInterpolator(
            (self.x_grid, self.y_grid), 
            density_field,
            method='linear',
            bounds_error=False,
            fill_value=1e-10
        )
        return interp_fn(self.Y)



class Sampler:
   
    @staticmethod

    
    def random_sampling(data, sampling_rate):
        
        
        n = data.shape[0]
       
        m = round(n * sampling_rate)
        
        perm = np.random.permutation(n)
        
        selected_indexes = perm[:m]
        
        return selected_indexes

    @staticmethod

    
    def blue_noise_sampling(data, sampling_rate, failure_tolerance=1000):
        
        
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
        
        
        n = data.shape[0]
        
        m = round(n * sampling_rate)
    
        selected_indexes = []

        
        first_choice = random.randint(0, n - 1)
        
        selected_indexes.append(first_choice)
        
        count = 1

        dist = cdist(np.array([data[first_choice]]), data, metric="euclidean").reshape(-1)

       
        while count < m:

            
            next_choice = dist.argmax()
            
            selected_indexes.append(next_choice)

            
            new_dist = cdist(np.array([data[next_choice]]), data, metric="euclidean").reshape(-1)

            
            dist = np.minimum(dist, new_dist)

            
            count += 1
            
        selected_indexes = np.array(selected_indexes)

        
        return selected_indexes

    @staticmethod

    
    def svd_based_sampling(data, sampling_rate):
        
        
        n = data.shape[0]
        
        m = round(n * sampling_rate)

        
        u, s, vt = np.linalg.svd(data.T, full_matrices=True)

        
        corr = np.sum(vt**2, axis=0)

        
        selected_indexes = corr.argsort()[-m:]

        
        return selected_indexes

    @staticmethod

    
    def hashmap_based_sampling(data, sampling_rate, grid_size=20, threshold=10):
        
        
        n = data.shape[0]
        
        m = round(n * sampling_rate)

        
        data = np.clip(data, 0, 1)

        
        discrete_full_data = np.clip((data * grid_size).astype(np.int64), 0, grid_size - 1)

        
        frozen = np.zeros((grid_size, grid_size)).astype(np.int64)
        
        counter = np.zeros((grid_size, grid_size)).astype(np.int64)

        
        perm = np.random.permutation(n)

        
        count = 0
        
        selected_idx = []
        
        for idx in perm:
            
            i, j = discrete_full_data[idx]
            
            counter[i][j] += 1
            
            if not frozen[i][j]:
               
                
                selected_idx.append(idx)
                
                count += 1
                
                
                if i > 0 and j > 0:
                    frozen[i - 1][j - 1] = 1
                
                if i > 0 and j < grid_size - 1:
                    frozen[i - 1][j + 1] = 1
                
                if i < grid_size - 1 and j > 0:
                    frozen[i + 1][j - 1] = 1
                
                if i < grid_size - 1 and j < grid_size - 1:
                    frozen[i + 1][j + 1] = 1
                
                
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
                    replace=False,
                    p=None if len(nonzero_indices) == 0 else None  
                )
            
            selected = np.concatenate([selected, supplement])
        
        return selected
    
    
    @staticmethod
    
    
    def density_blue_noise_sampling(data, density, sampling_rate, failure_tolerance=1000):
        
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
        
        n = data.shape[0]
        
        m = round(n * sampling_rate)
        
                
        if np.all(density == 0):
            density = np.ones(n)
        
        else:
            density = (density - np.min(density)) / (np.max(density) - np.min(density) + 1e-8)
        
               
        first_choice = np.argmax(density)
        
        selected_indexes = [first_choice]
        
                
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
                
        n = data.shape[0]
        
        m = round(n * sampling_rate)
        
        
        u, s, vt = np.linalg.svd(data.T, full_matrices=True)
        
               
        corr = np.sum(vt ** 2, axis=0) * density
        
        selected_indexes = corr.argsort()[-m:]
        
        return selected_indexes
    
    
    @staticmethod
        
    def density_hashmap_based_sampling(data, density, sampling_rate, grid_size=20, threshold=10):
                
        n = data.shape[0]
        
        m = round(n * sampling_rate)
        
        data = np.clip(data, 0, 1)
        
        discrete_data = (data * grid_size).astype(np.int64)
        
                
        discrete_data = np.clip(discrete_data, 0, grid_size - 1)
        
        frozen = np.zeros((grid_size, grid_size), dtype=int)
        
        counter = np.zeros((grid_size, grid_size), dtype=int)
        
                
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
                
                                
                if i > 0 and j > 0:
                    frozen[i-1][j-1] = 1
                
                if i > 0 and j < grid_size - 1:
                    frozen[i-1][j+1] = 1
                
                if i < grid_size - 1 and j > 0:
                    frozen[i+1][j-1] = 1
                
                if i < grid_size - 1 and j < grid_size - 1:
                    frozen[i+1][j+1] = 1
                    
                                
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


class PipelineController:
    def __init__(self, config):
        
        self.config = config
        self.loader = DataLoader()
        
    def run_full_pipeline(self):
        
        
        Y, y_add_list, y_sub_list, y_file = self.loader.load_data(
            
            self.config['num'], 
            
            self.config['filename'], 
            
            self.config['method'], 
            
            self.config['n_samples'],
            
            self.config['yita'],
            
            self.config['nbrs_k'],
            
            self.config['perplexity']
        )
        
        processor = GeometryProcessor(Y, y_add_list, y_sub_list)
        
        G = processor.compute_geometry_matrix()
                
        density_gen = DensityFieldGenerator(Y, G)
        
        density_field = density_gen.generate_density_field()
        
        point_density = density_gen.compute_point_density(density_field)
        
        
        os.makedirs("F:/sampling_results", exist_ok=True)
        
        densityfilename = y_file[:-4] + '_point_density.csv'
       
        np.savetxt(os.path.join("F:/sampling_results", densityfilename), point_density, delimiter=",")

        sampler = Sampler()
               
        for sample_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        
            
            standard_results = {
                
                'blue_noise': sampler.blue_noise_sampling(Y, sample_rate),
                
                'random': sampler.random_sampling(Y, sample_rate),
                
                'farthest': sampler.farthest_point_sampling(Y, sample_rate),
                
                'svd': sampler.svd_based_sampling(Y, sample_rate),
                
                'hash': sampler.hashmap_based_sampling(Y, sample_rate)
            }
            
            
            density_results = {
                
                'density_blue_noise': sampler.density_blue_noise_sampling(Y, point_density, sample_rate),
                
                'density_random': sampler.density_random_sampling(Y, point_density, sample_rate),
                
                'density_farthest': sampler.density_farthest_point_sampling(Y, point_density, sample_rate),
                
                'density_svd': sampler.density_svd_based_sampling(Y, point_density, sample_rate),
                
                'density_hash': sampler.density_hashmap_based_sampling(Y, point_density, sample_rate)
            }
            
           
            self._save_results(Y, standard_results, "standard", y_file, sample_rate, point_density)
            
            self._save_results(Y, density_results, "density_aware", y_file, sample_rate, point_density)
                    
            std_results = standard_results
                                            
            plt.figure(figsize=(16, 8))
                        
            plt.subplot(2, 5, 1)
            plt.scatter(Y[std_results['blue_noise']][:,0], Y[std_results['blue_noise']][:,1], s=15)
            plt.title("Standard Blue Noise")
            
            plt.subplot(2, 5, 2)
            plt.scatter(Y[std_results['random']][:,0], Y[std_results['random']][:,1], s=15)
            plt.title("Standard Random")
            
            plt.subplot(2, 5, 3)
            plt.scatter(Y[std_results['farthest']][:,0], Y[std_results['farthest']][:,1], s=15)
            plt.title("Standard Farthest")
            
            plt.subplot(2, 5, 4)
            plt.scatter(Y[std_results['svd']][:,0], Y[std_results['svd']][:,1], s=15)
            plt.title("Standard SVD")
            
            
            plt.subplot(2, 5, 5)
            plt.scatter(Y[std_results['hash']][:,0], Y[std_results['hash']][:,1], s=15)
            plt.title("Standard Hash")
                       
            
            plt.subplot(2, 5, 6)
            plt.scatter(Y[density_results['density_blue_noise']][:,0], Y[density_results['density_blue_noise']][:,1], s=15)
            plt.title("Density Blue Noise")
                                   
            plt.subplot(2, 5, 7)
            plt.scatter(Y[density_results['density_random']][:,0], Y[density_results['density_random']][:,1], s=15)
            plt.title("Density Random")
            
            
            plt.subplot(2, 5, 8)
            plt.scatter(Y[density_results['density_farthest']][:,0], Y[density_results['density_farthest']][:,1], s=15)
            plt.title("Density Farthest")
            
            
            plt.subplot(2, 5, 9)
            plt.scatter(Y[density_results['density_svd']][:,0], Y[density_results['density_svd']][:,1], s=15)
            plt.title("Density SVD")
            
            
            plt.subplot(2, 5, 10)
            plt.scatter(Y[density_results['density_hash']][:,0], Y[density_results['density_hash']][:,1], s=15)
            plt.title("Density Hash")
            
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
            
            plt.savefig("F:/sampling_results/" + y_file[:-4] + '_' + str(sample_rate) + "_comparison.png")
            plt.show()
            
            plt.close()
        
       
    def _save_results(self, data, results, prefix, y_file, sample_rate, point_density):
        
               
        for strategy, samples in results.items():
                       
            index_file = np.hstack((data[samples], samples.reshape(-1, 1)))
            
            index_label_file = np.hstack((index_file, label[samples]))
            
            index_label_filename = y_file[:-4] + '_' + f"{prefix}_{strategy}_index_label_samples" + '_' + str(sample_rate) + '.csv'
            
            np.savetxt(os.path.join("F:/sampling_results", index_label_filename), index_label_file, delimiter=",")


if __name__ == "__main__":
    
    num_samples = 10000

    label = y_MNIST[:num_samples, :]      
      
    config = {
        
        'num': 0,   
              
        'filename': 'MNIST',      
        
        'p_num': 4,               
        
        'yita': 'y05',             
        
        'method': 't-SNE',  
        
        'n_samples': 'n' + str(num_samples),        
        
        'nbrs_k': 'k20',            
    
        'perplexity': 'p30'      
    }
    
    controller = PipelineController(config)
    
    controller.run_full_pipeline()
                
                
                
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    