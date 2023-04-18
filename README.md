# API Wrapper for local LLMs
API wrapper that makes local LLMs compatible with OpenAI compatible clients (e.g., BabyAGI, LangeChain)
This is suitable for developers who want to build in-house services (since ChatGPT API is expensive)


## Installation

Use the package manager [conda](https://conda.io/projects/conda/en/latest/index.html) to install the required parameters

```bash
conda env create -f environment.yml
```

Run `setup.py` to put the project path to the PYTHONPATH

```bash
pip install -e .
```

### Usage

#### Step 1: Setting up the API server

By default the APIs will be accessed via "http://0.0.0.0:8000/v1/". The documentation of the APIs is accessed via "http://0.0.0.0:8000/v1/docs#"

```
python -m src.server.__main__ MODELNAME --model_path PATH_TO_MODEL_WEIGHT --lora_path PATH_TO_LORA_WEIGHT
```

where:
- MODELNAME: the name of the currently supported models. Currently the project only support Huggingface models. Possible model names are:
1. [llama](https://huggingface.co/docs/transformers/main/model_doc/llama). With this model, you can load the weights of the recent LLMs that are finetuned from LLama (Koala, Vicuna, Alpaca, ...). An example of using alpaca model finetuned with GPT-4.
```
python -m src.server.__main__  llama --model_path chavinlo/gpt4-x-alpaca
```


2. [alpacalora](https://github.com/tloen/alpaca-lora). This is specifically loading the AlpacaLora model. Example of using this model:

```
python -m src.server.__main__  llama --model_path decapoda-research/llama-13b-hf --lora_path chansung/alpaca-lora-13b
```

3. huggingface: Generic huggingface language models. It may take a while for huggingface to load this model. 



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

The example of using this APIService with Langchain BabyAGI is provided in the `example/` folder

### Documentation
The proper documation will be written soon.

### Resources
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

### License
This project is licensed under the terms of the MIT license.