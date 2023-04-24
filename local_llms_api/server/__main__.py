### This  implementation is inspired by python-llama-cpp (https://github.com/abetlen/llama-cpp-python)

import os
import json
from typing import List, Optional, Literal, Union, Iterator, Dict
from typing_extensions import TypedDict

from local_llms_api.server.llms import create_model, create_embedding_model
import local_llms_api.server.llms as llms

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings, Field, create_model_from_typeddict
from sse_starlette.sse import EventSourceResponse

def main(model, model_weight, lora_weight="", load8bit=False, separate_embedding=False, embedding_weight="intfloat/e5-base"):
    app = FastAPI(
        title="API Wrapper for Local LLMs",
        version="0.0.1",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    embedding_model = None
    if separate_embedding:
        embedding_model = create_embedding_model(embedding_weight, load8bit)
    model = create_model(model, model_weight, lora_weight, load8bit=load8bit)

    class CreateCompletionRequest(BaseModel):
        prompt: Union[str, List[str]]
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
        seed: int = -1
        add_bos_token: bool = True
        truncation_length: int = 2048
        ban_eos_token: bool = False
        skip_special_tokens: bool = True
        use_cache: bool = True
        
        
        logprobs: Optional[int] = Field(None)
        presence_penalty: Optional[float] = 0
        frequency_penalty: Optional[float] = 0
        best_of: Optional[int] = 1
        logit_bias: Optional[Dict[str, float]] = Field(None)
        user: Optional[str] = Field(None)

        class Config:
            schema_extra = {
                "example": {
                    "prompt": ["What is the capital of France?"],
                    "stop": ["\n", "###"],
                }
            }


    CreateCompletionResponse = create_model_from_typeddict(llms.Completion)

    @app.post(
        "/v1/completions",
        response_model=CreateCompletionResponse,
    )
    def create_completion(request: CreateCompletionRequest):
        if isinstance(request.prompt, list):
            request.prompt = "".join(request.prompt)

        completion_or_chunks = model(
            **request.dict(
                exclude={
                    "model",
                    "n",
                    "logprobs",
                    "frequency_penalty",
                    "presence_penalty",
                    "best_of",
                    "logit_bias",
                    "user",
                }
            )
        )
        if request.stream:
            chunks: Iterator[llms.CompletionChunk] = completion_or_chunks  # type: ignore
            return EventSourceResponse(dict(data=json.dumps(chunk)) for chunk in chunks)
        
        print(completion_or_chunks)
        completion: llms.Completion = completion_or_chunks  # type: ignore
        return completion

    class CreateEmbeddingRequest(BaseModel):
        input: List[str]
        model: Optional[str]
        user: Optional[str]
        encoding_format: Optional[str]

        class Config:
            schema_extra = {
                "example": {
                    "input": ["The food was delicious and the waiter..."],
                }
            }


    CreateEmbeddingResponse = create_model_from_typeddict(llms.Embedding)
    @app.post(
        "/v1/embeddings",
        response_model=CreateEmbeddingResponse,
    )
    def create_embedding(request: CreateEmbeddingRequest):
        if not separate_embedding:
            return model.create_embedding(**request.dict(exclude={"model", "user", "encoding_format"}))
        else:
            return embedding_model.create_embedding(**request.dict(exclude={"model", "user", "encoding_format"}))
        
    class ChatCompletionRequestMessage(BaseModel):
        role: Union[Literal["system"], Literal["user"], Literal["assistant"]]
        content: str
        user: Optional[str] = None

    class CreateChatCompletionRequest(BaseModel):
        model: Optional[str]
        messages: List[ChatCompletionRequestMessage]
        temperature: float = 0.8
        top_p: float = 0.95
        stream: bool = False
        stop: List[str] = []
        max_tokens: int = 128
        num_return_sequences: int=1
        output_scores: bool=False
        repetition_penalty: float=1.2

        # ignored or currently unsupported
        model: Optional[str] = Field(None)
        n: Optional[int] = 1
        presence_penalty: Optional[float] = 0
        frequency_penalty: Optional[float] = 0
        logit_bias: Optional[Dict[str, float]] = Field(None)
        user: Optional[str] = Field(None)

        class Config:
            schema_extra = {
                "example": {
                    "messages": [
                        ChatCompletionRequestMessage(
                            role="user", content="What is the capital of France?"
                        ),
                    ]
                }
            }


    CreateChatCompletionResponse = create_model_from_typeddict(llms.ChatCompletion)


    @app.post(
        "/v1/chat/completions",
        response_model=CreateChatCompletionResponse,
    )
    async def create_chat_completion(
        request: CreateChatCompletionRequest,
    ):
        completion_or_chunks = model.create_chat_completion(
            **request.dict(
                exclude={
                    "model",
                    "n",
                    "presence_penalty",
                    "frequency_penalty",
                    "logit_bias",
                    "user",
                }
            ),
        )

        if request.stream:

            async def server_sent_events(
                chat_chunks,
            ):
                for chat_chunk in chat_chunks:
                    yield dict(data=json.dumps(chat_chunk))
                yield dict(data="[DONE]")

            chunks = completion_or_chunks  # type: ignore

            return EventSourceResponse(
                server_sent_events(chunks),
            )
        completion = completion_or_chunks  # type: ignore
        return completion
    
    return app



if __name__ == "__main__":
    import os
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model to be used")
    parser.add_argument("--model_path", help="Path to the model weight")
    parser.add_argument("--lora_path", help="Path to the lora weight if the model use lora weight")
    parser.add_argument("--load8bit", help="Whether to load 8 bit", action='store_true', default=False)
    parser.add_argument("--separate_embedding", help="Whether to load another model for embedding", action='store_true', default=False)
    parser.add_argument("--embedding_path", help="Path to the additional embedding model", default='intfloat/e5-base')

    args = parser.parse_args()
    
    app = main(args.model, args.model_path, args.lora_path, args.load8bit, args.separate_embedding, args.embedding_path)

    uvicorn.run(
        app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000))
    )

