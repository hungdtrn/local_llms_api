import json
from urllib import response
import requests
import json
import openai

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

            

class LLMService:
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
    
    def create_completion(self, prompt, suffix=None, max_tokens=16,
                          temperature=0.9, top_p=0.95, echo=False,
                          stop=[], stream=False, repeat_penalty=1.5,
                          top_k=40, **kwargs):
        if self.llm_type == "local":
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
                "top_k": top_k,
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
                                            repeat_penalty=repeat_penalty)
            
    
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
                "top_k": top_k,
                **kwargs
            }))
                        
            return converto_to_response_obj({"response": response.json()})
        elif self.llm_type == "openai":
            return converto_to_response_obj({"response":openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens)})
            
        