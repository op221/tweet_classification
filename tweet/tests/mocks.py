from typing import List
import numpy as np

class MockModel:
    def __init__(self, random_state=None, 
                    autotune = None,
                    batch_size = None, test_size=None):
        pass

    def load_df(self, filepath, index_col, text_col, target_col):
        pass

    def build_model(self,
                    preprocess_handle=None,
                    encoder_handle=None,
                    drop_out=None):
        self.model = None
    
    def train(self, num_epochs = None, lr=None, optimizer_name=None):
        self.history = None
        return self.history
    
    def save(self, model_path=None):
        pass

    def reload(self, model_path=None):
        self.model = None
    
    def predict(self, inputs:List[str]) -> List[int]:
        return list(np.random.randint(2, size=len(inputs)))

    def evaluate(self, test_ds):
        pass