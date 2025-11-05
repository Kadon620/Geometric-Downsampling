import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from tqdm import tqdm
from .local_pca import LocalPCAAnalyzer


class T_SNE:
    """t-SNE降维实现"""
    
    def __init__(self, n_component=2, perplexity=100.0):
        self.n_component = n_component
        self.perplexity = perplexity
        self.beta = None
        self.kl = []
        self.final_kl = None
        self.final_iter = 0
        self.P = None
        self.P0 = None
        self.Q = None

    def Hbeta(self, D=np.array([]), beta=1.0):
        """计算Hbeta值"""
        P = np.exp(-D.copy() * beta)
        sum_p = sum(P)
        H = np.log(sum_p) + beta * np.sum(D * P) / sum_p
        P = P / sum_p
        return H, P

    def x2p(self, X=np.array([]), tol=1e-12, perplexity=100.0):
        """计算概率矩阵P"""
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        P = np.zeros((n, n))
        beta = np.ones((n, 1)) / np.max(D)
        logU = np.log(perplexity)

        for i in range(n):
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
            (H, thisP) = self.Hbeta(Di, beta[i])

            Hdiff = H - logU
            tries = 0
            while np.abs(Hdiff) > tol and tries < 1000:
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                (H, thisP) = self.Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

        self.beta = beta
        return P

    def fit_transform(self, X, max_iter=1000, early_exaggerate=True, y_init=None):
        """执行t-SNE降维"""
        (n, d) = X.shape
        no_dims = self.n_component
        initial_momentum = 0.5
        final_momentum = 0.8
        if not early_exaggerate:
            final_momentum = 0.0
        eta = 500
        min_gain = 0.01

        # 初始化
        if y_init is None:
            Y = np.random.randn(n, no_dims)
        else:
            Y = y_init
            
        dY = np.zeros((n, no_dims))
        iY = np.zeros((n, no_dims))
        gains = np.ones((n, no_dims))

        # 计算P矩阵
        P = self.x2p(X, 1e-15, self.perplexity)
        self.P0 = P.copy()
        P = P + np.transpose(P)
        P = P / (2*n)
        P = np.maximum(P, 1e-120)
        self.P = P.copy()

        if early_exaggerate:
            P = P * 4.

        # 优化过程
        for iter in range(max_iter):
            sum_Y = np.sum(np.square(Y), 1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-120)

            PQ = P - Q
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

            if iter < 20 and early_exaggerate:
                momentum = initial_momentum
            else:
                momentum = final_momentum
                
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.))
            if not early_exaggerate:
                gains[gains < min_gain] = min_gain
                
            iY = momentum * iY - (eta / (np.sqrt(0.1*iter+0.1))) * (gains * dY)
            Y = Y + iY

            if iter == 100 and early_exaggerate:
                P = P / 4.

        # 计算最终的Q矩阵
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-120)
        self.Q = Q

        return Y


class DRTrans:
    """降维变换基类"""
    
    def __init__(self):
        self.X = None
        self.Y = None
        self.label = None
        self.derivative = None
        self.point_error = None
        self.y_add_list = []
        self.y_sub_list = []
        self.k = 0
        self.eigen_number = 2
        self.linearity = None

    def transform(self, nbrs_k, MAX_EIGEN_COUNT=5, yita=0.3):
        """执行变换（子类需要实现）"""
        return self.Y, self.y_add_list, self.y_sub_list


class PCATrans(DRTrans):
    """PCA降维变换"""
    
    def __init__(self, X, label=None):
        super().__init__()
        self.X = X
        self.label = np.ones((X.shape[0], 1)) if label is None else label
        self.n_samples = X.shape[0]
        self.Y = None
        self.y_add = None
        self.y_sub = None
        self.y_add_list = []
        self.y_sub_list = []
        self.derivative = None
        self.point_error = None
        self.k = 0
        self.eigen_number = 2

    def transform_(self, eigen_weights, eigen_vectors_list, yita):
        """执行PCA变换"""
        n, dim = self.X.shape
        y_add_list = []
        y_sub_list = []
        
        for loop_index in range(self.eigen_number):
            eigenvectors = eigen_vectors_list[loop_index]
            x_add_v = np.zeros((n, dim))
            x_sub_v = np.zeros((n, dim))
            
            for i in range(n):
                x_add_v[i, :] = self.X[i, :] + yita * eigen_weights[i, loop_index] * eigenvectors[i, :]
                x_sub_v[i, :] = self.X[i, :] - yita * eigen_weights[i, loop_index] * eigenvectors[i, :]
            
            y_add_v = np.matmul(x_add_v, self.derivative)
            y_sub_v = np.matmul(x_sub_v, self.derivative)
            y_add_list.append(y_add_v)
            y_sub_list.append(y_sub_v)
        
        return y_add_list, y_sub_list
    
    def transform(self, nbrs_k, MAX_EIGEN_COUNT=5, yita=0.3):
        """执行PCA降维和变换"""
        n, dim = self.X.shape
        
        if MAX_EIGEN_COUNT > dim:
            print("Warning: the input MAX_EIGEN_COUNT is too large.")
            MAX_EIGEN_COUNT = dim
        
        self.k = nbrs_k
        self.eigen_number = MAX_EIGEN_COUNT
        
        # 局部PCA分析
        analyzer = LocalPCAAnalyzer(n_neighbors=nbrs_k, max_eigen_count=MAX_EIGEN_COUNT)
        eigen_vectors_list, eigen_weights = analyzer.analyze(self.X)

        print("Computing dimension reduction...")
        
        # PCA降维
        pca = PCA(n_components=2, copy=True, whiten=True)
        pca.fit(self.X)
        P_ = pca.components_
        P = np.transpose(P_)
        self.Y = np.matmul(self.X, P)
        self.derivative = P
        
        # 执行变换
        self.y_add_list, self.y_sub_list = self.transform_(eigen_weights, eigen_vectors_list, yita)

        return self.Y, self.y_add_list, self.y_sub_list


class MDSTrans(DRTrans):
    """MDS降维变换"""
    
    def __init__(self, X, label=None, y_init=None, y_precomputed=False):
        super().__init__()
        self.X = X
        self.label = np.ones((X.shape[0], 1)) if label is None else label
        self.n_samples = X.shape[0]
        self.Y = None
        self.Y_add = None
        self.Y_sub = None
        self.y_add_list = []
        self.y_sub_list = []
        self.derivative = None
        self.point_error = None
        self.k = 0
        self.init_y(y_init, y_precomputed)

    def init_y(self, y_init, y_precomputed):
        """初始化MDS降维结果"""
        print("Computing dimension reduction...")
        
        if y_precomputed and (y_init is not None):
            mds = MDS(n_components=2, n_init=1, max_iter=1000, eps=0.01)
            Y = mds.fit_transform(self.X, init=y_init)
        else:
            print("使用默认的MDS初始化方式")
            mds = MDS(n_components=2, max_iter=1000, eps=0.01)
            Y = mds.fit_transform(self.X)
            
        print('MDS降维结束，开始向量变换')
        self.Y = Y

    def transform(self, nbrs_k, MAX_EIGEN_COUNT=5, yita=0.3):
        """执行MDS变换"""
        print("Transform MDS...")
        n, dim = self.X.shape
        
        if MAX_EIGEN_COUNT > dim:
            print("Warning: the input MAX_EIGEN_COUNT is too large.")
            MAX_EIGEN_COUNT = dim
        
        self.k = nbrs_k
        self.eigen_number = MAX_EIGEN_COUNT
        
        # 局部PCA分析
        analyzer = LocalPCAAnalyzer(n_neighbors=nbrs_k, max_eigen_count=MAX_EIGEN_COUNT)
        eigen_vectors_list, eigen_weights = analyzer.analyze(self.X)
        
        print("Computing vectors transformation...")
        
        # 计算距离矩阵
        self.Dx = euclidean_distances(self.X)
        self.Dy = euclidean_distances(self.Y)
        
        # 导入向量变换（避免循环导入）
        from .vector_transforms import MDS_Vector_Transform_Optimized_kuai
        
        # 执行向量变换
        transformer = MDS_Vector_Transform_Optimized_kuai(self.X, self.Y, self.Dx, self.Dy)
        self.y_add_list, self.y_sub_list = transformer.transform_all_kuai(eigen_vectors_list, yita * eigen_weights)
    
        return self.Y, self.y_add_list, self.y_sub_list


class TSNETrans(DRTrans):
    """t-SNE降维变换"""
    
    def __init__(self, X, label=None, y_init=None, perplexity=100.0):
        super().__init__()
        self.X = X
        self.label = np.ones((X.shape[0], 1)) if label is None else label
        self.n_samples = X.shape[0]
        self.Y = None
        self.y_add = None
        self.y_sub = None
        self.y_add_list = []
        self.y_sub_list = []
        self.P0 = None
        self.Px = None
        self.Q = None
        self.beta = None
        self.derivative = None
        self.point_error = None
        self.k = 0
        self.eigen_number = 2
        self.init_y(y_init, perplexity)

    def init_y(self, Y0, perplexity=100.0):
        """初始化t-SNE降维结果"""
        print("Computing dimension reduction...")
        
        t_sne = T_SNE(perplexity=perplexity)
        
        if Y0 is None:
            Y = t_sne.fit_transform(self.X, max_iter=1000)
        else:
            Y = t_sne.fit_transform(self.X, max_iter=1000, early_exaggerate=False, y_init=Y0)
            
        print('t-SNE降维结束，开始向量变换')
        self.Y = Y
        self.beta = t_sne.beta
        self.P0 = t_sne.P0
        self.Px = t_sne.P
        self.Q = t_sne.Q

    def transform(self, nbrs_k, MAX_EIGEN_COUNT=5, yita=0.3):
        """执行t-SNE变换"""
        print("Transform t-SNE...")
        n, dim = self.X.shape
        
        if MAX_EIGEN_COUNT > dim:
            print("Warning: the input MAX_EIGEN_COUNT is too large.")
            MAX_EIGEN_COUNT = dim
        
        self.k = nbrs_k
        self.eigen_number = MAX_EIGEN_COUNT
        
        # 局部PCA分析
        analyzer = LocalPCAAnalyzer(n_neighbors=nbrs_k, max_eigen_count=MAX_EIGEN_COUNT)
        eigen_vectors_list, eigen_weights = analyzer.analyze(self.X)
        
        # 计算距离矩阵
        self.Dx = euclidean_distances(self.X)
        self.Dy = euclidean_distances(self.Y)
        
        # 导入向量变换（避免循环导入）
        from .vector_transforms import TSNE_Optimized_Vector_Transform_kuai
        
        # 执行向量变换
        transformer = TSNE_Optimized_Vector_Transform_kuai(self.X, self.Y, self.Dy, self.beta, self.P0, self.Q)
        self.y_add_list, self.y_sub_list = transformer.transform_all_kuai(eigen_vectors_list, yita * eigen_weights)

        return self.Y, self.y_add_list, self.y_sub_list