from ..custom_llm import CustomLLM
from dotenv import load_dotenv
import os

load_dotenv()
host = os.getenv("HOST", "http://0.0.0.0:8000/v1")

llm = CustomLLM(host=host, temperature=0.7)
print(llm("What is the capital of Australia?"))
