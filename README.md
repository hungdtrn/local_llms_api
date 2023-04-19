# API Wrapper for local LLMs
API wrapper that makes local LLMs compatible with OpenAI compatible clients (e.g., BabyAGI, LangeChain)
This is suitable for developers who want to build in-house LLM API services (since ChatGPT API is expensive).



## Installation

Use the package manager [conda](https://conda.io/projects/conda/en/latest/index.html) to install the required parameters

```bash
conda create -n local_llms_api python=3.9
conda activate local_llms_api
pip install -r requirement.txt
```

Run `setup.py` to put the project path to the PYTHONPATH

```bash
pip install -e .
```

### Usage

#### Step 1: Setting up the API server
By default the APIs will be accessed via "http://0.0.0.0:8000/v1/". The documentation of the APIs is accessed via "http://0.0.0.0:8000/v1/docs#"

```
python -m local_llms_api.server MODELNAME --model_path PATH_TO_MODEL_WEIGHT --lora_path PATH_TO_LORA_WEIGHT
```

MODELNAME: the name of the currently supported models. Currently the project only support Huggingface models. The supported models are:
1. [llama](https://huggingface.co/docs/transformers/main/model_doc/llama). With this model, you can load the weights of the recent LLMs that are finetuned from LLama (Koala, Vicuna, Alpaca, ...). An example of using alpaca model finetuned with GPT-4.
2. [alpacalora](https://github.com/tloen/alpaca-lora). This is specifically loading the AlpacaLora model. Example of using this model:
3. huggingface: Generic huggingface language models. It may take a while for huggingface to load this model. 

Example
```
# llama
python -m local_llms_api.server  llama --model_path chavinlo/gpt4-x-alpaca

# Alpaca lora
python -m local_llms_api.server  llama --model_path decapoda-research/llama-13b-hf --lora_path chansung/alpaca-lora-13b --load8bit

# Local model (e.g., vicuna weights)
python -m local_llms_api.server  llama --model_path path_to_vicuna_weight

```

#### Step 2: Use the APIService
Example of using the APIService

```
from local_llms_api import LLMService

local_llm = LLMService(host="http://0.0.0.0:8000/v1")

# Sentence embedding:
embedding = local_llm.create_embedding(["Hello"]).data[0].embedding

# Sentence completion
completion = local_llm.create_completion(prompt="Hello, How are you?", max_tokens=128, temperature=1.0).response.choices[0].text.strip()

# Chat completion
chat_completion = local_llm.create_chat_completion(messages= [{"role": "user", "content": "What is the weather today?"}], max_tokens=512, top_p=1, temperature=1).response.choices[0].message.content.strip()
```

**Note:** If you host the API server at a different machine with a different address, you need to replace "http://0.0.0.0:8000/v1" with your address.

### Example of using with LangChain
The example of using this APIService with Langchain BabyAGI is provided in the `example/` folder. 

```
# Install extra requirement for the Langchain example
pip install -r example/extra_requirement.txt

# Add env file
cp example/langchain/.env.example example/langchain/.env

# UPDATE FILE .env, add your SERPAPI_API_KEY and HOST

# Run the example code
python -m example.langchain.simple_example
```

### Documentation
The proper documation will be written soon.

### Resources
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python): The implementation of the API is motivated by this repo

### License
This project is licensed under the terms of the MIT license.