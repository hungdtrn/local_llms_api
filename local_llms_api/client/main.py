import json
from typing import List
from urllib import response
import requests
import json
import openai

class ResponseObj(dict):
    def __init__(self, resp: dict) -> None:
        for k, v in resp.items():
            self[k] = converto_to_response_obj(v)
        
    def __getattr__(self, k):
        return self[k]

def converto_to_response_obj(resp):
    if isinstance(resp, dict) and not isinstance(resp, ResponseObj):
        return ResponseObj(resp)
    elif isinstance(resp, list):
        return [converto_to_response_obj(x) for x in resp]
    else:
        return resp

            

class LLMClient:
    def __init__(self, host="http://0.0.0.0:8000/v1", llm_type="local", openai_model="gpt-3.5-turbo") -> None:
        if llm_type == "local":
            self.host = host
        elif llm_type == "openai":
            self.caller = openai
            self.model = openai_model
        
        self.llm_type = llm_type
        
    def create_embedding(self, input, **kwargs):
        if self.llm_type == "local":
            response = requests.post(self.host + "/embeddings", data=json.dumps({
                "input": input,
            }))
            return converto_to_response_obj(response.json())
        elif self.llm_type == "openai":
            return openai.Embedding.create(input=input, model="text-embedding-ada-002")
    
    def create_completion(self, prompt, max_tokens: int = 16,
                            do_sample: bool = True,
                            temperature: float = 0.8,
                            top_p: float = 0.95,
                            echo: bool = False,
                            stop: List[str] = [],
                            stream: bool = False,
                            num_return_sequences: int=1,
                            output_scores: bool=False,
                            repetition_penalty: float=1.2,
                            top_k: int = 1,
                            num_beams: int = 1,
                            seed: int = -1,
                            add_bos_token: bool = True,
                            truncation_length: int = 2048,
                            ban_eos_token: bool = False,
                            skip_special_tokens: bool = True,
                            use_cache: bool = True,
                            **kwargs):
        
        if self.llm_type == "local":
            if stop is None:
                stop = []
            response = requests.post(self.host + "/completions", data=json.dumps({
                "prompt": prompt,
                "do_sample": do_sample,
                "num_return_sequences": num_return_sequences,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "echo": echo, "stop": stop,
                "stream": stream, "repetition_penalty": repetition_penalty,
                "output_scores": output_scores,
                "seed": seed,
                "add_bos_token": add_bos_token,
                "truncation_length": truncation_length,
                "ban_eos_token": ban_eos_token,
                "skip_special_tokens": skip_special_tokens,
                "use_cache": use_cache,
                **kwargs
            }))
            
            if response.status_code != 200:
                print(response.json())
                raise Exception(response)
            else:
                return converto_to_response_obj({"response": response.json()})
        elif self.llm_type == "openai":
            return openai.Completion.create(engine=self.model,
                                            prompt=prompt,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
                                            top_p=top_p,
                                            top_k=top_k,
                                            repeat_penalty=repetition_penalty)
            
    
    def create_chat_completion(self, messages, suffix=None, max_tokens=16,
                          temperature=0.8, top_p=0.95, echo=False,
                          stop=[], stream=False, repeat_penalty=1.1,
                          top_k=40, **kwargs):
        if self.llm_type == "local":
            response = requests.post(self.host + "/chat/completions", data=json.dumps({
                "messages": messages,
                "suffix": suffix,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "echo": echo, "stop": stop,
                "stream": stream, "repeat_penalty": repeat_penalty,
                **kwargs
            }))
                        
            return converto_to_response_obj({"response": response.json()})
        elif self.llm_type == "openai":
            return converto_to_response_obj({"response":openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens)})
            
        