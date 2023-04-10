from llama_cpp import Llama

from .base import Base

class Llamacpp(Base):
    def __init__(self, model_path, max_tokens, num_threads=8) -> None:
        super().__init__()
        
        self.model = Llama(model_path=model_path, embedding=True, n_threads=num_threads)
        self.max_tokens = max_tokens
        
    def get_embedding(self, text):
        return self.model.embed(text)