from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from typing import Optional, List, Mapping, Any
from local_llms_api.service import LLMService

class CustomLLM(LLM):
    host: str="http://0.0.0.0:8000/v1"
    temperature: float=0.7
    max_new_tokens: int=256
    repeat_penalty: float=1.1
    top_p: float=1
    top_k: float=40

    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is None:
            stop = []
                        
        return LLMService(host=self.host).create_chat_completion(messages=[{"role": "user", "content": prompt}], temperature=self.temperature, stop=stop,
                                         max_tokens=self.max_new_tokens, repeat_penalty=self.repeat_penalty,
                                         top_p=self.top_p, top_k=self.top_k).response.choices[0].message.content.strip()

        # return LLMService(host=self.host).create_completion(prompt, temperature=self.temperature, stop=stop,
        #                                  max_tokens=self.max_new_tokens, repeat_penalty=self.repeat_penalty,
        #                                  top_p=self.top_p, top_k=self.top_k).response.choices[0].text.strip()

        
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"host": self.host,
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "repeat_penalty": self.repeat_penalty,
                "top_p": self.top_p,
                "top_k": self.top_k}
        

class CustomEmbeddings(Embeddings):
    def __init__(self, host) -> None:
        super().__init__()
        self.host = host

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [x.embedding for x in LLMService(self.host).create_embedding(input=texts).data]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return LLMService(self.host).create_embedding(input=[text]).data[0].embedding

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"host": self.host}
