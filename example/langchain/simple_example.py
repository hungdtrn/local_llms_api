from ..custom_llm import CustomLLM
from dotenv import load_dotenv
import os

load_dotenv()
host = "http://a100-2.ai.deakin.edu.au:8000/v1"

llm = CustomLLM(host=host, temperature=0.7, output_scores=True, num_return_sequences=10,
                top_p=1.5, do_sample=True, repetition_penalty=1.2, cache=False)
print(llm("Hello"))
