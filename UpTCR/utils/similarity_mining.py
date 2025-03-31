from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseSimilarityGenerator(nn.Module, ABC):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    @abstractmethod
    def generate_with_param(self, params: List):
        pass

    @abstractmethod
    def get_indexed_parameters(self, indices=None) -> List:
        pass

    def load_params(self, params):
        raise NotImplementedError("")

class FullSimilarityGenerator(BaseSimilarityGenerator):
    def __init__(self, dim):
        super().__init__(dim)
        self.sim_mat = nn.Parameter(torch.eye(dim))
            
    def generate_with_param(self, params):
        assert len(params) == 1
        return params[0]

    def get_indexed_parameters(self, indices=None) -> List:
        if indices is None:
            sim = self.sim_mat
        else:
            sim = self.sim_mat[indices[:,None], indices]
        return [sim]
    
    def load_params(self, params):
        s = params[0]
        self.sim_mat.data = s