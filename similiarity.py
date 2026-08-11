import json
import os
from dotenv import load_dotenv

from scipy.spatial import distance
from openai import OpenAI
print(distance.cosine([0, 1], [1, 0]))

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    response_dict = response.model_dump()
    return [data['embedding'] for data in response_dict['data']]


with open("products.json") as f:
    products = json.load(f)


list_of_descriptions = ['Charge your devices conveniently with this sleek wireless charging dock.', 'Elevate your skincare routine with this luxurious skincare set.']


# Embed short_description and print
print(create_embeddings(products[0]['short_description'])[0])

# Embed list_of_descriptions and print
print(create_embeddings(list_of_descriptions))