import os
import sys
import json
import uuid
import time
from typing import List, Optional, Literal, Union, Iterator, Dict
import gc
from accelerate import dispatch_model

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings, Field, create_model_from_typeddict
from sse_starlette.sse import EventSourceResponse

import torch
from torch import Tensor
import transformers
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, StoppingCriteria as BaseStoppingCriteria, StoppingCriteriaList, AutoModel
from peft import PeftModel
import numpy as np
from local_llms_api.server.llms.base import Base
from .prompter import Prompter, Conversation, turn_to_message_array
# Import math Library
import math 

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
        
    def __is_stop(self, input_ids, stop):
        # count = int(len(stop) * 4 / 5)
        count = len(stop) - 1
        count = count if count > 0 else 0
        if torch.sum(input_ids[0, -len(stop):] == stop) > count:
            print(input_ids[0, -len(stop):], stop)
            return True
        
        return False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop in self.stops:
            if len(stop) == 0:
                continue
            
            if self.__is_stop(input_ids, stop):
                return True
            
        return False
    
    def remove_stop_from_input(self, input_ids: torch.LongTensor):
        for stop in self.stops:
            if len(stop) == 0:
                continue
            
            if self.__is_stop(input_ids, stop):
                return input_ids[:, :-len(stop)]
        
        return input_ids


class LLMModel(Base):
    def __init__(self, model_name: str, model_path: str, lora_path: str, load_in_8bit: bool) -> None:
        if model_name == "llama":
            tokenizer = LlamaTokenizer.from_pretrained(model_path, 
                                                       device_map="auto",
                                                       load_in_8bit=load_in_8bit)
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                load_in_8bit=load_in_8bit
            )
            self.embed_tokens = model.model.embed_tokens    
        elif model_name == "alpacalora":
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
                
            self.embed_tokens = model.model.model.embed_tokens    
            
        elif model_name == "huggingface":
            tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto",)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
            )
            self.embed_tokens = model.model.embed_tokens    

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

            
    def encode(self, text, add_special_tokens=True, add_bos_token=True, truncation_length=None):
        # Tokenize the prompt
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
        input_ids = inputs["input_ids"]
        
        if not add_bos_token and input_ids[0][0] == self.tokenizer.bos_token_id:
            input_ids = input_ids[:, 1:]

        # Llama adds this extra token when the first character is '\n', and this
        # compromises the stopping criteria, so we just remove it
        if type(self.tokenizer) is transformers.LlamaTokenizer and input_ids[0][0] == 29871:
            input_ids = input_ids[:, 1:]

        # Handling truncation
        if truncation_length is not None:
            input_ids = input_ids[:, -truncation_length:]

        input_ids = inputs["input_ids"].to(device)
        
        return input_ids
   
    def create_completion(
        self,
        prompt: str,
        stop: List[str] = [],
        stream: bool = False,
        max_tokens: int = 128,
        add_special_tokens: bool = True,
        add_bos_token: bool = True,
        truncation_length: int = 2048,
        echo: bool = False,
        seed: int = -1,
        ban_eos_token: bool = True,
        skip_special_tokens: bool = True,
        **kwargs
    ):
        completion_id = f"cmpl-{str(uuid.uuid4())}"
        created = int(time.time())

        # Prepare the prompt
        prompt = self.prompter.generate_prompt(prompt)
        
        input_ids = self.encode(prompt, add_special_tokens=add_special_tokens, 
                                add_bos_token=add_bos_token, truncation_length=truncation_length)
        
        # Prepare the stopping criteria
        stops = [self.encode(word, add_special_tokens=False)[0] for word in stop]
        
        stopping_criteria = StoppingCriteriaList([StoppingCriteria(stops=stops)])        
        
        generation_config = {
            "return_dict_in_generate": True,
            "stopping_criteria": stopping_criteria,
            "max_new_tokens": max_tokens + input_ids.shape[-1],
        }
        
        if ban_eos_token:
            generation_config["suppress_tokens"] = [self.tokenizer.eos_token_id]
        
        generation_config.update(kwargs)
                
        if stream:
            pass
        
        # Without streaming
        print(generation_config)
        with torch.no_grad():
            generation_output = self.model.generate(input_ids=input_ids, **generation_config)
        
        promptlen = input_ids.shape[-1]
        generated_sequences = generation_output.sequences

        output_sequence, start_idx = list(zip(*[self.prompter.get_response(x) for x in self.tokenizer.batch_decode(generated_sequences, skip_special_tokens=skip_special_tokens)]))
        scores = generation_output.scores
        
        if scores is None:
            logprobs = [None for i in range(len(output_sequence))]
        else:
            vocab_log_probs = torch.stack(scores, dim=1).log_softmax(-1)  # [n, length, vocab_size]
            token_log_probs = torch.gather(vocab_log_probs, 2, generated_sequences[:, promptlen:, None]).squeeze(-1).tolist()  # [n, length]
            logprobs = [np.mean(token_log_probs[i]) for i in range(kwargs['num_return_sequences'])]
            logprobs = [score if not math.isinf(score) else -99999 for score in logprobs]
        
        prompt_len = input_ids.shape[-1]
        out_len = [x.shape[-1] for x in generated_sequences]
        del input_ids
        del generation_output
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": self.model_name,
            "choices": [
                {
                    "text": output,
                    "index": i,
                    "logprob":  score,
                    "finish_reason": None,
                }
            for i, (output, score) in enumerate(zip(output_sequence, logprobs))],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": [x for x in out_len],
                "total_tokens": [prompt_len + x for x in out_len],
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
        output_scores: bool=False,
        repetition_penalty: float=1.2,
        num_return_sequences: int=1

    ):
        
        messages.append({"role": "assistant", "content": None})
        conversation = Conversation(messages=turn_to_message_array(messages)) 
        PROMPT = conversation.get_prompt()
        # PROMPT_STOP = ["###", "\nuser: ", "\nassistant: ", "\nsystem: "]
        PROMPT_STOP = []
        
        completion_or_chunks = self.create_completion(prompt=PROMPT,
                                                      stop=PROMPT_STOP + stop,
                                                      temperature=temperature,
                                                      top_p=top_p,
                                                      top_k=top_k,
                                                      stream=stream,
                                                      max_tokens=max_tokens,
                                                      output_scores=output_scores,
                                                      repetition_penalty=repetition_penalty,
                                                      num_return_sequences=num_return_sequences)
        
        if stream:
            chunks = completion_or_chunks
            return self._convert_text_completion_chunks_to_chat(chunks)
        else:
            completion = completion_or_chunks
            return self._convert_text_completion_to_chat(completion)
        

class EmbeddingModel:
    def __init__(self, model_path: str, load8bit: bool) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, load_in_8bit=load8bit, device_map='auto')
        self.model = AutoModel.from_pretrained(model_path, load_in_8bit=load8bit).to("cuda")
        
        self.model = dispatch_model(self.model, device_map='auto')

        self.model_path = model_path

    def average_pool(self, last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def create_embedding(self, input: List[str]):
        batch_dict = self.tokenizer(input, return_tensors="pt", padding=True)
        for k in batch_dict.keys():
            batch_dict[k] = batch_dict[k].to(self.model.device)

        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = embeddings.detach().cpu().tolist()

        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": x,
                    "index": i,
                } for i, x in enumerate(embeddings)
            ],
            "model": self.model_path,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }
        
