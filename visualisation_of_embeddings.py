import json
import os

from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


with open("products.json") as f:
    products = json.load(f)

# print(products)

product_descriptions = [product['short_description'] for product in products]


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client. embeddings.create(
model="text-embedding-3-small",
input=product_descriptions
)

response_dict = response.model_dump()
print('************ Response Dict ************')
print(response_dict)
print('************ END Response Dict ************')
# Extract the embeddings from response_dict and store in products
for i, product in enumerate(products):
    product['embedding'] = response_dict['data'][i]['embedding']
    
# print(products[0].items())

# Create reviews and embeddings lists using list comprehensions
categories = [product['category'] for product in products]
embeddings = [product['embedding'] for product in products]

# Reduce the number of embeddings dimensions to two using t-SNE
tsne = TSNE(n_components=2, perplexity=5)
embeddings_2d = tsne.fit_transform(np.array(embeddings))


# Plot the 2D embeddings
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1])
for i, category in enumerate(categories):
    plt.annotate(category, (embeddings_2d[i, 0], embeddings_2d[i, 1]))


plt.show()