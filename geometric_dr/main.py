import numpy as np
import os
import warnings
from data_loader import DataLoader
from dr_methods import PCATrans, MDSTrans, TSNETrans

warnings.filterwarnings("ignore")


def run_example():
    """运行示例"""
    # 配置参数
    nbrs_k = 20
    MAX_EIGEN_COUNT = 4
    yita = 0.5
    
    # 加载数据
    data_loader = DataLoader()
    X, label = data_loader.load_mnist(num_samples=10000)
    
    for method in ('MDS', 't-SNE', 'PCA'):
        print(f"正在进行 {method} 方法处理")
        
        n, d = X.shape
        print(f"样本总共选择 {n} 个")

        # 选择降维方法
        if method == 'MDS':
            print("实例化MDSTrans对象")
            trans = MDSTrans(X, label=label, y_init=None, y_precomputed=False)
            
        elif method == 't-SNE':
            print("实例化TSNETrans对象")
            trans = TSNETrans(X, label=label, y_init=None, perplexity=100.0)
            
        elif method == 'PCA':
            print("实例化PCATrans对象")
            trans = PCATrans(X, label=label)
        
        else:
            print(f"当前不支持该方法: {method}")
            return

        # 执行降维转换
        print("对数据进行降维转换")
        trans.transform(nbrs_k=nbrs_k, MAX_EIGEN_COUNT=MAX_EIGEN_COUNT, yita=yita)
        
        # 保存结果
        save_results(trans, method, n, nbrs_k)


def save_results(trans, method, n, nbrs_k):
    """保存结果到文件"""
    base_path = "F:/data_new/dr_data_mnist/"
    os.makedirs(base_path, exist_ok=True)
    
    # 保存降维结果
    np.savetxt(f"{base_path}MNIST_{method}_{n}_{nbrs_k}_Y.csv", trans.Y, fmt='%.18e', delimiter=",")
    
    # 保存变换结果
    for i in range(len(trans.y_add_list)):
        np.savetxt(f"{base_path}MNIST_{method}_{n}_{nbrs_k}_Y_add_list{i}.csv", 
                   trans.y_add_list[i], fmt='%.18e', delimiter=",")
        np.savetxt(f"{base_path}MNIST_{method}_{n}_{nbrs_k}_Y_sub_list{i}.csv", 
                   trans.y_sub_list[i], fmt='%.18e', delimiter=",")


if __name__ == '__main__':
    run_example()