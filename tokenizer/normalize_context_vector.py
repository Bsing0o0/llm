import torch
import torch.nn.functional as F

embedding = torch.tensor(

    [[0.43, 0.12, 0.78],  # your (X^1)
     [0.56, 0.34, 0.91],  # journey (X^2)
     [0.67, 0.45, 0.78],  # starts (X^3)
     [0.23, 0.56, 0.34],  # with (X^4)
     [0.89, 0.67, 0.12],  # one (X^5)
     [0.45, 0.78, 0.23]]  # step (X^6)
)

#Dot Product Attention Calculation

query = embedding[1]  # "journey"

att_score = torch.empty(embedding.shape[0])

for i, x_1 in enumerate(embedding):
    att_score[i] = torch.dot( x_1, query ) # dot product between the query and each token embedding
print("Attention scores for 'journey':", att_score)

# Normalize the attention scores to get the attention weights, which helps us understand how much attention the model should pay to each token when processing "journey". 
# The weights are calculated by dividing each attention score by the sum of all attention scores, ensuring that the weights sum up to 1.

# softmax function

# We can use this in other files as well, so it's a good idea to define it as a separate function.


def get_attention_weights(att_score):
    return F.softmax(att_score, dim=0)

att_weight = get_attention_weights(att_score)

print("Attention weights for 'journey':", att_weight)
print("Sum of attention weights:", att_weight.sum())  # should be 1.0

def context_vector(att_weight, embedding):
    context_vec = torch.zeros(embedding.shape[1])
    
    for i, x_1 in enumerate(embedding):
        context_vec += att_weight[i] * x_1
        
    return context_vec

context_vec = context_vector(att_weight, embedding)
print("Context vector for 'journey':", context_vec)

#Calculate for all tokens in the input sequence
context_vectors = torch.empty(embedding.shape)
for i in range(embedding.shape[0]):   # Loop through each token in the input sequence
    query = embedding[i]              
    att_score = torch.empty(embedding.shape[0])        # Calculate attention scores for the current query token against all tokens in the input sequence
    for j, x_1 in enumerate(embedding):                # Loop through each token in the input sequence to calculate the attention score (dot product) between the query token and each token in the input sequence
        att_score[j] = torch.dot(x_1, query)       # Store the attention score for the current query token and each token in the input sequence
    att_weight = get_attention_weights(att_score)  # Normalize the attention scores to get the attention weights for the current query token
    context_vec = context_vector(att_weight, embedding)            # Calculate the context vector for the current query token using the attention weights and the input token embeddings
    context_vectors[i] = context_vec
print("Context vectors for all tokens:\n", context_vectors)

