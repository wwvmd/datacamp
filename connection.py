import os

from dotenv import load_dotenv
from openai import OpenAI

PRICING_PER_1M = {
    # Pricing shown as USD per 1M tokens.
    "text-embedding-3-small": {
        "input": 0.02,
    }
}

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client. embeddings.create(
model="text-embedding-3-small",
input="Embeddings are a numerical representation of text that can be used to measure the relatedness between two pieces of text."
)

response_dict = response.model_dump()
embedding_dim = len(response_dict["data"][0]["embedding"])
print(f"Embedding generated. Dimensions: {embedding_dim}")
usage = response.usage
input_tokens = getattr(usage, "input_tokens", None)
if input_tokens is None:
    input_tokens = getattr(usage, "prompt_tokens", 0)

rates = PRICING_PER_1M["text-embedding-3-small"]
cost = (input_tokens / 1_000_000) * rates["input"]

print(f"Estimated cost: ${cost:.6f}")