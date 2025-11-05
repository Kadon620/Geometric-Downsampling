Geometric Downsampling (GDS)
https://img.shields.io/badge/python-3.8+-blue.svg
https://img.shields.io/badge/license-MIT-green.svg

Official implementation of "Geometric Downsampling: High-Dimensional Local Feature-Guided Sampling in Projection Space".

This repository provides a novel framework that systematically integrates high-dimensional local geometry into the low-dimensional sampling process, addressing the geometric distortion inherent in conventional "reduce-then-sample" pipelines.

🚀 核心特性
几何感知采样: 首个将高维局部几何系统集成到低维采样过程中的方法

多降维方法支持: 支持 PCA、MDS、t-SNE 等多种降维方法

多种采样策略: 提供蓝噪声、最远点、SVD、哈希等多种采样方法

密度场引导: 基于几何特征的密度场指导采样过程

大规模优化: 分块计算策略支持大规模数据处理

📁 项目结构
text
Geometric-Downsampling/
├── 📁 geometric_dr/              # 几何降维模块
│   ├── data_loader.py           # 数据加载
│   ├── local_pca.py             # 局部PCA分析
│   ├── dr_methods.py            # 降维方法 (PCA, MDS, t-SNE)
│   ├── mds_vector_transforms.py # MDS向量变换
│   ├── tsne_vector_transforms.py # t-SNE向量变换
│   └── main.py                  # 降维主程序
├── 📁 density_sampling/         # 密度采样模块
│   ├── data_loader.py           # 数据加载
│   ├── knn_utils.py             # K近邻工具
│   ├── geometry_processor.py    # 几何处理器
│   ├── density_field.py         # 密度场生成
│   ├── sampling_methods.py      # 采样方法
│   ├── visualization.py         # 可视化
│   ├── pipeline_controller.py   # 流水线控制器
│   └── density_sampling_main.py # 采样主程序
├── 📁 examples/                 # 使用示例
├── 📁 data/                     # 数据目录
├── 📁 results/                  # 结果输出
├── requirements.txt             # 依赖列表
└── README.md                    # 项目说明

🛠️ 安装与依赖
环境要求
Python 3.8+

CUDA (可选，用于GPU加速)

安装依赖
bash
pip install -r requirements.txt
依赖包
text
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pandas>=1.3.0
torch>=1.9.0
tqdm>=4.62.0
numba>=0.55.0
ucimlrepo>=0.0.3
torchvision>=0.10.0
cffi>=1.15.0

📖 快速开始
1. 几何降维 (第一步)
python
from geometric_dr.data_loader import DataLoader
from geometric_dr.dr_methods import PCATrans, MDSTrans, TSNETrans

# 加载数据
loader = DataLoader()
X, labels = loader.load_mnist(num_samples=10000)

# 选择降维方法 (PCA, MDS, 或 t-SNE)
trans = TSNETrans(X, labels)
Y, y_add_list, y_sub_list = trans.transform(
    nbrs_k=20, 
    MAX_EIGEN_COUNT=4, 
    yita=0.5
)

# 保存结果
np.savetxt("results/t-SNE_embedding.csv", Y, delimiter=",")
2. 密度采样 (第二步)
python
from density_sampling.pipeline_controller import PipelineController

# 配置参数
config = {
    'num': 0,
    'filename': 'MNIST',
    'method': 't-SNE',
    'n_samples': 'n10000',
    'yita': 'y05',
    'nbrs_k': 'k20',
    'perplexity': 'p30'
}

# 运行完整流水线
controller = PipelineController(config)
controller.run_full_pipeline()
3. 完整流程示例
python
import numpy as np
from geometric_dr.data_loader import DataLoader as DRDataLoader
from geometric_dr.dr_methods import TSNETrans
from density_sampling.pipeline_controller import PipelineController

# 步骤1: 降维
dr_loader = DRDataLoader()
X, labels = dr_loader.load_mnist(10000)

# 执行t-SNE降维
tsne = TSNETrans(X, labels)
Y, y_add_list, y_sub_list = tsne.transform(nbrs_k=20, MAX_EIGEN_COUNT=4, yita=0.5)

# 步骤2: 采样
config = {
    'num': 0,
    'filename': 'MNIST',
    'method': 't-SNE', 
    'n_samples': 'n10000',
    'yita': 'y05',
    'nbrs_k': 'k20',
    'perplexity': 'p30'
}

controller = PipelineController(config)
sampling_results = controller.run_full_pipeline()

🎯 核心算法
几何降维流程
局部PCA分析: 对每个点的邻域进行PCA，提取局部几何特征

降维映射: 使用PCA、MDS或t-SNE进行降维

向量变换: 将高维局部特征投影到低维空间

密度采样流程
几何矩阵计算: 基于扰动向量计算局部几何特征

密度场生成: 构建几何感知的密度场

自适应采样: 基于密度场进行几何感知采样

📊 支持的数据集
MNIST: 手写数字数据集

合成数据: 可控的高斯分布数据

Mice Protein: 蛋白质表达数据集

Ecoli Proteins: 大肠杆菌蛋白质数据集

🔧 配置参数
降维参数
python
config = {
    'nbrs_k': 20,           # K近邻数量
    'MAX_EIGEN_COUNT': 4,   # 最大特征数量
    'yita': 0.5,            # 扰动系数
    'perplexity': 30        # t-SNE困惑度
}
采样参数
python
sampling_rates = [0.1, 0.2, 0.3, 0.4, 0.5]  # 采样率
grid_size = 200                              # 密度场网格大小
bandwidth_scale = 0.5                       # 带宽缩放因子
📈 实验结果
根据论文实验，GDS方法在多个指标上显著优于传统方法：

Neighbor Hit (NH): 提升达 7.7%

Trustworthiness (TW): 一致改善

QNX/RNX: 局部结构保持更好

用户偏好: 在视觉评估中获得显著偏好

🎨 可视化
项目提供完整的可视化功能，包括：

降维结果可视化

采样结果对比

密度场可视化

聚类边界展示

🔬 引用
如果您在研究中使用了本代码，请引用我们的论文：

bibtex
@article{xiang2025geometric,
  title={Geometric Downsampling: High-Dimensional Local Feature-Guided Sampling in Projection Space},
  author={Xiang, Xudong and Qin, Hongxing and Hu, Haibo and Xiang, Tao and Chen, Baoquan},
  journal={arXiv preprint},
  year={2025}
}
🤝 贡献
欢迎提交 Issue 和 Pull Request 来改进项目！

📄 许可证
本项目采用 MIT 许可证 - 详见 LICENSE 文件。

📞 联系方式
作者: Xudong Xiang

邮箱: d220201045@stu.cqupt.edu.cn

项目地址: https://github.com/kadon620/Geometric-Downsampling

如果这个项目对您有帮助，请给我们一个 ⭐️ ！
