class Base:
    def __init__(self) -> None:
        pass
    
    def create_embedding(self, *args, **kwargs):
        pass
    
    def create_completion(self, *args, **kwargs):
        pass
    
    def create_chat_completion(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.create_completion(*args, **kwargs)