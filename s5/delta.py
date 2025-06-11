from flax import linen as nn
import jax.numpy as np
from jax.nn.initializers import lecun_normal

class Delta(nn.Module):
    R: int  # Delta의 차원
    
    def setup(self):
        # R 차원의 맥락 정보를 생성하는 행렬
        self.context_matrix = self.param("Delta",
                                       lecun_normal(),
                                       (self.R, self.R))
        
    def __call__(self, q):
        """
        Args:
            q: 현재 상태 (R 차원)
        Returns:
            p: A에게 전달할 맥락 정보 (R 차원)
        """
        # R 차원의 상태를 R 차원의 맥락 정보로 변환
        p = self.context_matrix @ q
        return p 