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

plt.title('Token Embeddings in 3D Space')  # Set the title of the plot
plt.show()   # Display the 3D scatter plot of token embeddings


#Dot Product Attention Calculation

query = input[1]  # "journey"

att_score_2 = torch.empty(input.shape[0])

for i, x_1 in enumerate(input):
    att_score_2[i] = torch.dot( x_1, query ) # dot product between the query and each token embedding
print("Attention scores for 'journey':", att_score_2)

# Normalize the attention scores to get the attention weights, which helps us understand how much attention the model should pay to each token when processing "journey". 
# The weights are calculated by dividing each attention score by the sum of all attention scores, ensuring that the weights sum up to 1.

#there are different ways to normalize the attention scores.

#1. division by the sum of attention scores
att_weights = att_score_2 / torch.sum(att_score_2)


print("Attention weights for 'journey':", att_weights)
print ("Sum of attention weights:", torch.sum(att_weights))  # Should be 1.0    

#2. softmax function
att_weights_softmax = torch.nn.functional.softmax(att_score_2, dim=0)


print("Attention weights for 'journey' (softmax):", att_weights_softmax)
print("Sum of attention weights (softmax):", torch.sum(att_weights_softmax))  # Should be 1.0