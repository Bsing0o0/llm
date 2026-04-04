import torch

input = torch.tensor(

    [[0.43, 0.12, 0.78],  # your (X^1)
     [0.56, 0.34, 0.91],  # journey (X^2)
     [0.67, 0.45, 0.78],  # starts (X^3)
     [0.23, 0.56, 0.34],  # with (X^4)
     [0.89, 0.67, 0.12],  # one (X^5)
     [0.45, 0.78, 0.23]]  # step (X^6)
)

# Just a simple 3D scatter plot to visualize the token embeddings in 3D space.
# Each point corresponds to a token embedding, and the coordinates are the values of the embedding dimensions. 
# In this case, we are using only 3 dimensions for visualization purposes, but in practice, the embedding dimension can be much higher (e.g., 256 or 512).

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

words = ["your", "journey", "starts", "with", "one", "step"]
x = input[:, 0].numpy()
y = input[:, 1].numpy()
z = input[:, 2].numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for x, y, z, word in zip(x, y, z, words):
    ax.scatter(x, y, z)
    ax.text(x, y, z, word, fontsize=10)


ax.set_xlabel('Embedding Dimension 1')
ax.set_ylabel('Embedding Dimension 2')
ax.set_zlabel('Embedding Dimension 3')

plt.title('Token Embeddings in 3D Space')
plt.show()
