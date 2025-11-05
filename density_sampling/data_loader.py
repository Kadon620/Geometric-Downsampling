import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DataLoader:
    """数据加载器"""
    
    def __init__(self, base_path="F:/dr_data_mnist_kuai"):
        self.base_path = base_path
    
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