import numpy as np
from data_loader import DataLoader
from pipeline_controller import PipelineController


if __name__ == "__main__":
    
    # 加载MNIST数据
    num_samples = 10000
    loader = DataLoader()
    X_MNIST, y_MNIST = loader.load_mnist(num_samples)
    
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
    
    # 运行流水线
    controller = PipelineController(config)
    controller.run_full_pipeline()