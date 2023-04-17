import json
from urllib import response
import requests
import json

class ResponseObj(dict):
    def __init__(self, resp: dict) -> None:
        for k, v in resp.items():
            self.__setattr__(k, converto_to_response_obj(v))
    
    def __setattr__(self, __name: str, __value) -> None:
        return super().__setattr__(__name, __value)
    
    def __getattr__(self, k):
        return self[k]

def converto_to_response_obj(resp):
    if isinstance(resp, dict) and not isinstance(resp, ResponseObj):
        return ResponseObj(resp)
    elif isinstance(resp, list):
        return [converto_to_response_obj(x) for x in resp]
    else:
        return resp

            

class LLMCaller:
    def __init__(self, host="http://0.0.0.0/v1") -> None:
        self.host = host
        
    def create_embedding(self, input):
        
        response = requests.post(self.host + "/embeddings", data=json.dumps({
            "input": input,
        }))
        
        return converto_to_response_obj(response.json())
    
    def create_completion(self, prompt, suffix=None, max_tokens=16,
                          temperature=0.9, top_p=0.95, echo=False,
                          stop=[], stream=False, repeat_penalty=1.5,
                          top_k=40):
        if stop is None:
            stop = []
        response = requests.post(self.host + "/completions", data=json.dumps({
            "prompt": prompt,
            "suffix": suffix,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "echo": echo, "stop": stop,
            "stream": stream, "repeat_penalty": repeat_penalty,
            "top_k": top_k
        }))
        
        if response.status_code != 200:
            print(response.json())
            raise Exception(response)
        else:
            return converto_to_response_obj({"response": response.json()})
    
    def create_chat_completion(self, prompt, suffix=None, max_tokens=16,
                          temperature=0.8, top_p=0.95, echo=False,
                          stop=[], stream=False, repeat_penalty=1.1,
                          top_k=40):
        response = requests.post(self.host + "/completions", data={
            "prompt": prompt,
            "suffix": suffix,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "echo": echo, "stop": stop,
            "stream": stream, "repeat_penalty": repeat_penalty,
            "top_k": top_k
        })
        
        return converto_to_response_obj({"response": response.json()})

        
        