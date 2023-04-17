import os
import json
from typing import List, Optional, Literal, Union, Iterator, Dict
from typing_extensions import TypedDict

from src.api.llms import create_model
import src.api.llms as llms

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings, Field, create_model_from_typeddict
from sse_starlette.sse import EventSourceResponse

def main(model, model_weight, lora_weight=""):
    app = FastAPI(
        title="🦙 llama.cpp Python API",
        version="0.0.1",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    model = create_model(model, model_weight, lora_weight)

    class CreateCompletionRequest(BaseModel):
        prompt: Union[str, List[str]]
        suffix: Optional[str] = Field(None)
        max_tokens: int = 16
        temperature: float = 0.8
        top_p: float = 0.95
        echo: bool = False
        stop: List[str] = []
        stream: bool = False

        # ignored or currently unsupported
        model: Optional[str] = Field(None)
        n: Optional[int] = 1
        logprobs: Optional[int] = Field(None)
        presence_penalty: Optional[float] = 0
        frequency_penalty: Optional[float] = 0
        best_of: Optional[int] = 1
        logit_bias: Optional[Dict[str, float]] = Field(None)
        user: Optional[str] = Field(None)

        # llama.cpp specific parameters
        top_k: int = 40
        repeat_penalty: float = 1.1

        class Config:
            schema_extra = {
                "example": {
                    "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
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
                    "input": "The food was delicious and the waiter...",
                }
            }


    CreateEmbeddingResponse = create_model_from_typeddict(llms.Embedding)
    @app.post(
        "/v1/embeddings",
        response_model=CreateEmbeddingResponse,
    )
    def create_embedding(request: CreateEmbeddingRequest):
        return model.create_embedding(**request.dict(exclude={"model", "user", "encoding_format"}))

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

        # ignored or currently unsupported
        model: Optional[str] = Field(None)
        n: Optional[int] = 1
        presence_penalty: Optional[float] = 0
        frequency_penalty: Optional[float] = 0
        logit_bias: Optional[Dict[str, float]] = Field(None)
        user: Optional[str] = Field(None)

        # llama.cpp specific parameters
        repeat_penalty: float = 1.1

        class Config:
            schema_extra = {
                "example": {
                    "messages": [
                        ChatCompletionRequestMessage(
                            role="system", content="You are a helpful assistant."
                        ),
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
    parser.add_argument("--model_weight", help="Path to the model weight")
    parser.add_argument("--lora_weight", help="Path to the lora weight if the model use lora weight")
    args = parser.parse_args()
    
    app = main(args.model, args.model_weight, args.lora_weight)

    uvicorn.run(
        app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000))
    )

