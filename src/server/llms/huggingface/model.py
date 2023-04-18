import os
import sys
import json
import uuid
import time
from typing import List, Optional, Literal, Union, Iterator, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings, Field, create_model_from_typeddict
from sse_starlette.sse import EventSourceResponse

import torch
import transformers
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, StoppingCriteria as BaseStoppingCriteria, StoppingCriteriaList
from peft import PeftModel

from src.server.llms.base import Base
from .prompter import Prompter, Conversation, turn_to_message_array

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass
        
class StoppingCriteria(BaseStoppingCriteria):
    def __init__(self, stops=[]) -> None:
        super().__init__()
        self.stops = stops
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop in self.stops:
            if len(stop) == 0:
                continue
            
            count = int(len(stop) * 2 / 3)
            if torch.sum(input_ids[0, -len(stop):] == stop) > count:
                return True
            
            # if torch.any(input_ids[0, -len(stop):] == stop):
            #     return True
            
        return False
    
    def remove_stop_from_input(self, input_ids: torch.LongTensor):
        for stop in self.stops:
            if len(stop) == 0:
                continue
            
            count = int(len(stop) * 2 / 3)
            if torch.sum(input_ids[0, -len(stop):] == stop) > count:
                return input_ids[:, :-len(stop)]
        
        return input_ids

class Model(Base):
    def __init__(self, model_name: str, model_path: str, lora_path: str, load_in_8bit: bool) -> None:
        if model_name == "llama":
            tokenizer = LlamaTokenizer.from_pretrained(model_path, device_map="auto",)
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
            )
        elif model_name == "alpaca_lora":
            tokenizer = LlamaTokenizer.from_pretrained(model_path)
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=load_in_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            model = PeftModel.from_pretrained(
                model,
                lora_path,
                torch_dtype=torch.float16,
            )
            
            if not load_in_8bit:
                model.half()  # seems to fix bugs for some users.
        elif model_name == "huggingface":
            tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto",)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
            )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
            
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.embed_tokens = model.model.embed_tokens
        self.prompter = Prompter()
            
    def create_embedding(self, input: List[str]):
        tokens = self.tokenizer(input, return_tensors="pt", padding=True)
        tokens_id = tokens["input_ids"].to(device)
        embedding = self.embed_tokens(tokens_id).mean(-2)
        embedding = embedding.detach().cpu().tolist()
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": x,
                    "index": i,
                } for i, x in enumerate(embedding)
            ],
            "model": self.model_name,
            "usage": {
                "prompt_tokens": len(tokens),
                "total_tokens": len(tokens),
            },
        }
   
    def create_completion(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: List[str] = [],
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
    ):
        completion_id = f"cmpl-{str(uuid.uuid4())}"
        created = int(time.time())

        # Prepare the prompt
        prompt = self.prompter.generate_prompt(prompt)
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        # Prepare the stopping criteria
        stops = [self.tokenizer(word, return_tensors="pt")["input_ids"][0, 1:].to(device) for word in stop]
        stopping_criteria = StoppingCriteriaList([StoppingCriteria(stops=stops)])        
        
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repeat_penalty
        )
        
        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_tokens,
            "stopping_criteria": stopping_criteria,
        }
        
        if stream:
            pass
        
        # Without streaming
        with torch.no_grad():
            generation_output = self.model.generate(**generate_params)
        # Remove stop words
        if len(stops) is not None:
            generation_output.sequences = StoppingCriteria(stops=stops).remove_stop_from_input(generation_output.sequences)

        s = generation_output.sequences[0]
        
        output = self.tokenizer.decode(s)

        output = self.prompter.get_response(output)
                
        if echo:
            output = prompt + output
            
        if suffix is not None:
            output = output + suffix
        
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": self.model_name,
            "choices": [
                {
                    "text": output,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
            "usage": {
                "prompt_tokens": len(input_ids[0]),
                "completion_tokens": len(s),
                "total_tokens": len(input_ids[0]) + len(s),
            },
        }
    
    def _convert_text_completion_to_chat(
        self, completion
    ):
        return {
            "id": "chat" + completion["id"],
            "object": "chat.completion",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion["choices"][0]["text"],
                    },
                    "finish_reason": completion["choices"][0]["finish_reason"],
                }
            ],
            "usage": completion["usage"],
        }

    def _convert_text_completion_chunks_to_chat(
        self,
        chunks,
    ):
        for i, chunk in enumerate(chunks):
            if i == 0:
                yield {
                    "id": "chat" + chunk["id"],
                    "model": chunk["model"],
                    "created": chunk["created"],
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            yield {
                "id": "chat" + chunk["id"],
                "model": chunk["model"],
                "created": chunk["created"],
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk["choices"][0]["text"],
                        },
                        "finish_reason": chunk["choices"][0]["finish_reason"],
                    }
                ],
            }

    
    def create_chat_completion(
        self,
        messages: List[object],
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        stream: bool = False,
        stop: List[str] = [],
        max_tokens: int = 128,
        repeat_penalty: float = 1.1,
    ):
        
        messages.append({"role": "assistant", "content": None})
        conversation = Conversation(messages=turn_to_message_array(messages)) 
        PROMPT = conversation.get_prompt()
        # PROMPT_STOP = ["###", "\nuser: ", "\nassistant: ", "\nsystem: "]
        PROMPT_STOP = []
        print(PROMPT, PROMPT_STOP)

        completion_or_chunks = self.create_completion(prompt=PROMPT,
                                                      stop=PROMPT_STOP + stop,
                                                      temperature=temperature,
                                                      top_p=top_p,
                                                      top_k=top_k,
                                                      stream=stream,
                                                      max_tokens=max_tokens,
                                                      repeat_penalty=repeat_penalty,)
        
        if stream:
            chunks = completion_or_chunks
            return self._convert_text_completion_chunks_to_chat(chunks)
        else:
            completion = completion_or_chunks
            return self._convert_text_completion_to_chat(completion)