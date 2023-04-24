from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from typing import Optional, List, Mapping, Any
from local_llms_api.client import LLMClient

class CustomLLM(LLM):
    host: str="http://0.0.0.0:8000/v1"
    max_tokens: int = 16
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.95
    echo: bool = False
    stop: List[str] = []
    stream: bool = False
    num_return_sequences: int=1
    output_scores: bool=False
    repetition_penalty: float=1.2
    top_k: int = 1
    num_beams: int = 1
    seed: int = -1
    add_bos_token: bool = True
    truncation_length: int = 2048
    ban_eos_token: bool = False
    skip_special_tokens: bool = True
    use_cache: bool = True

    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is None:
            stop = []
                        
        # return LLMClient(host=self.host).create_chat_completion(messages=[{"role": "user", "content": prompt}], temperature=self.temperature, stop=stop,
        #                                  max_tokens=self.max_new_tokens, repeat_penalty=self.repeat_penalty,
        #                                  top_p=self.top_p, top_k=self.top_k).response.choices[0].message.content.strip()

        out = LLMClient(host=self.host).create_completion(prompt, 
                                                           max_tokens=self.max_tokens,
                                                           do_sample=self.do_sample,
                                                           temperature=self.temperature,
                                                           top_p=self.top_p,
                                                           echo=self.echo,
                                                           stop=self.stop,
                                                           stream=self.stream,
                                                           num_return_sequences=self.num_return_sequences,
                                                           output_scores=self.output_scores,
                                                           repetition_penalty=self.repetition_penalty,
                                                           top_k=self.top_k,
                                                           num_beams=self.num_beams,
                                                           seed=self.seed,
                                                           add_bos_token=self.add_bos_token,
                                                           truncation_length=self.truncation_length,
                                                           ban_eos_token=self.ban_eos_token,
                                                           skip_special_tokens=self.skip_special_tokens,
                                                           use_cache=self.use_cache).response.choices
        print(out)
        return out[0].text.strip()

        
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
        return [x.embedding for x in LLMClient(self.host).create_embedding(input=texts).data]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return LLMClient(self.host).create_embedding(input=[text]).data[0].embedding

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"host": self.host}
