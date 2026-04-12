import torch


input = torch.tensor(
    [[0.43, 0.12, 0.78],  # your (X^1)
     [0.56, 0.34, 0.91],  # journey (X^2)
     [0.67, 0.45, 0.78],  # starts (X^3)
     [0.23, 0.56, 0.34],  # with (X^4)
     [0.89, 0.67, 0.12],  # one (X^5)
     [0.45, 0.78, 0.23]]  # step (X^6)
    
)

# A = Second token embedding (journey)
#B = Embedding dimension (3 in this case)
#C = Desired output dimension for the query, key, and value vectors (e.g., 2)

x_2 = input [1] #A
d_in = input.shape[1] #B
d_out = 2 #C

torch.manual_seed(123)  # Set a random seed for reproducibility
W_q = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)  # Query weight matrix, requires_grad=False means that we won't be updating these weights during training, which is just for demonstration purposes. In a real model, these would be learnable parameters.
W_k = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)  # Key weight matrix
W_v = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)  # Value weight matrix


# Compute the query, key, and value vectors for the token "journey" using the respective weight matrices.
query = x_2 @ W_q # Q = X_2 * W_q
key = x_2 @ W_k    # K = X_2 * W_k
value = x_2 @ W_v  # V = X_2 * W_v

print("Query vector for 'journey':", query)
print ("Query Shape:", query.shape)  # Check the shape of the query vector to ensure it matches the expected output dimension (d_out)

print("Key vector for 'journey':", key) 
#print ("Key Shape:", key.shape)  # Check the shape of the key vector to ensure it matches the expected output dimension (d_out)

print("Value vector for 'journey':", value)
#print ("Value Shape:", value.shape)  # Check the shape of the value vector to ensure it matches the expected output dimension (d_out)

#attention Score Calculation

# To calculate the attention scores, we need to compute the dot product between the query vector of "journey" and the key vectors of all tokens in the input sequence.

keys = input @ W_k  # Compute the key vectors for all tokens in the input sequence by multiplying the input matrix with the key weight matrix. This will give us a matrix of shape (number of tokens, d_out), where each row corresponds to the key vector of a token.
print ("Key vectors for all tokens:", keys, "\n""\n")  # This will show us the key vectors for all tokens in the input sequence.

queries_2 = query @ keys.T #Transpose the key matrix to align dimensions for the dot product
print("Attention scores for 'journey':", queries_2)  # This will give us the attention scores for "journey" with respect to all tokens in the input sequence.

#Lets make a funtion call  to calculate key, query and value vectors for any token in the input sequence.
def compute_qkv(token_embedding, W_q, W_k, W_v):
    query = token_embedding @ W_q
    key = token_embedding @ W_k
    value = token_embedding @ W_v
    return query, key, value

