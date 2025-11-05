import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


class DataLoader:
    """数据加载器，支持多种数据集"""
    
    def __init__(self, base_path="F:/dr_data_mnist_kuai"):
        self.base_path = base_path
        self.label_encoder = LabelEncoder()
    
    def _construct_filename(self, num, filename, method, n_samples, yita, nbrs_k, perplexity, file_type, list_index=None):
        """构造文件名"""
        parts = [
            f"{num}_{filename}_{method}_{n_samples}_{yita}_{nbrs_k}_{perplexity}",
            "Y" if file_type == "Y" else 
            f"Y_{'add' if file_type.startswith('add') else 'sub'}_list{list_index}"
        ]
        return f"{'_'.join(parts)}.csv"

    def load_data(self, num, filename, method, n_samples, yita, nbrs_k, perplexity):
        """加载数据"""
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

    def load_mnist(self, num_samples=10000):
        """加载MNIST数据集"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

        train_dataset = datasets.MNIST(root='E:/Data/MNIST', train=True, transform=transform, download=True)
        
        X_MNIST = train_dataset.train_data
        X_MNIST = np.array(X_MNIST, dtype=np.float64)
        y_MNIST = train_dataset.train_labels
        y_MNIST = np.array(y_MNIST, dtype=np.float64)

        X_MNIST = X_MNIST[:num_samples].reshape(-1, 28 * 28)
        y_MNIST = y_MNIST[:num_samples].reshape(-1,)
        
        return X_MNIST, y_MNIST

    def load_synthetic_data(self, num_samples=500, num_features=64, overlap_factor=0.5):
        """生成合成数据"""
        np.random.seed(0)
        
        # 第一个高斯分布
        mean1 = np.zeros(num_features)
        cov1 = np.eye(num_features)
        data1 = np.random.multivariate_normal(mean1, cov1, num_samples)

        # 第二个高斯分布
        mean2 = overlap_factor * np.ones(num_features)
        cov2 = np.eye(num_features)
        data2 = np.random.multivariate_normal(mean2, cov2, num_samples)

        # 合并数据
        data = np.vstack([data1, data2])
        labels = np.hstack([np.zeros(num_samples), np.ones(num_samples)])
        
        return data, labels

    def load_mice_protein(self):
        """加载Mice Protein数据集"""
        mice_protein_expression = fetch_ucirepo(id=342)
        
        X_mice = mice_protein_expression.data.features
        X_mice_value = X_mice.iloc[:,:-3]
        X_mice_value.fillna(X_mice_value.mode().iloc[0], inplace=True)
        
        y_mice = mice_protein_expression.data.targets
        
        X_mice_array = np.array(X_mice_value, dtype=np.float64)
        y_mice_label = self.label_encoder.fit_transform(y_mice)
        y_mice_array = np.array(y_mice_label, dtype=np.float64)
        
        return X_mice_array, y_mice_array


def read_csv_files_in_folder(folder_path):
    """读取文件夹中的所有CSV文件"""
    dataframes_and_filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path, header=None)
                dataframes_and_filenames.append((data, file))
    return dataframes_and_filenames