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



# list_of_descriptions = ['Charge your devices conveniently with this sleek wireless charging dock.', 'Elevate your skincare routine with this luxurious skincare set.']


# # Embed short_description and print
# print(create_embeddings(products[0]['short_description'])[0])

# # Embed list_of_descriptions and print
# print(create_embeddings(list_of_descriptions))

product_to_find = "soap"
product_to_find_embedding = create_embeddings(product_to_find)[0]


for product in products:
    product['embedding'] = create_embeddings(product['short_description'])[0]

distances = []

for product in products:
    row = []
    row.append(distance.cosine(product['embedding'], product_to_find_embedding))
    distances.append(row)

import numpy as np
print(f"closest product to {product_to_find}:  {products[np.argmin([row[0] for row in distances])]['short_description']}");
