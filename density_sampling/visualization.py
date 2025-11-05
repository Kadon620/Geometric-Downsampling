import matplotlib.pyplot as plt
import numpy as np


def save_visualization(Y, standard_results, density_results, filename, sample_rate):
    """保存可视化结果"""
    plt.figure(figsize=(16, 8))
    
    # 标准采样方法
    plt.subplot(2, 5, 1)
    plt.scatter(Y[standard_results['blue_noise']][:,0], Y[standard_results['blue_noise']][:,1], s=15)
    plt.title("Standard Blue Noise")
    
    plt.subplot(2, 5, 2)
    plt.scatter(Y[standard_results['random']][:,0], Y[standard_results['random']][:,1], s=15)
    plt.title("Standard Random")
    
    plt.subplot(2, 5, 3)
    plt.scatter(Y[standard_results['farthest']][:,0], Y[standard_results['farthest']][:,1], s=15)
    plt.title("Standard Farthest")
    
    plt.subplot(2, 5, 4)
    plt.scatter(Y[standard_results['svd']][:,0], Y[standard_results['svd']][:,1], s=15)
    plt.title("Standard SVD")
    
    plt.subplot(2, 5, 5)
    plt.scatter(Y[standard_results['hash']][:,0], Y[standard_results['hash']][:,1], s=15)
    plt.title("Standard Hash")
    
    # 密度感知采样方法
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
    plt.savefig("F:/sampling_results/" + filename[:-4] + '_' + str(sample_rate) + "_comparison.png")
    plt.show()
    plt.close()