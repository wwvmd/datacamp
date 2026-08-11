import matplotlib.pyplot as plt

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

topics = [article['topic'] for article in articles]
for i, topic in enumerate(topics):
plt.annotate(topic, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.show()