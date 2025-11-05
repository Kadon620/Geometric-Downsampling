"""
Geometric Downsampling (GDS) - 几何下采样

一个用于高维数据几何感知下采样的Python库。
"""

__version__ = "0.1.0"
__author__ = "Xudong Xiang et al."

from .data_loader import DataLoader
from .local_pca import LocalPCAAnalyzer, local_pca_dn, KNN, normalize
from .dr_methods import DRTrans, PCATrans, MDSTrans, TSNETrans, T_SNE
from .mds_vector_transforms import MDS_Vector_Transform_Optimized_kuai
from .tsne_vector_transforms import TSNE_Optimized_Vector_Transform_kuai

__all__ = [
    'DataLoader',
    'LocalPCAAnalyzer',
    'local_pca_dn',
    'KNN',
    'normalize',
    'DRTrans',
    'PCATrans',
    'MDSTrans',
    'TSNETrans',
    'T_SNE',
    'MDS_Vector_Transform_Optimized_kuai',
    'TSNE_Optimized_Vector_Transform_kuai'
]