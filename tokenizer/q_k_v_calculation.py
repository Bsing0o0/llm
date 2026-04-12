#Calculate Q, K, V for a given input sequence
import torch
def compute_qkv(token_embedding, W_q, W_k, W_v):
    query = token_embedding @ W_q
    key = token_embedding @ W_k
    value = token_embedding @ W_v
    return query, key, value

#Attention Score Calculation
def calculate_attention_scores(query, keys):
    attention_scores = query @ keys.T  # Compute the dot product between the query vector and the key vectors of all tokens
    return attention_scores


#Test the functions with a sample input
input = torch.tensor(
    [[0.43, 0.12, 0.78],  # your (X^1)
     [0.56, 0.34, 0.91],  # journey (X^2)
     [0.67, 0.45, 0.78],  # starts (X^3)
     [0.23, 0.56, 0.34],  # with (X^4)
     [0.89, 0.67, 0.12],  # one (X^5)
     [0.45, 0.78, 0.23]]  # step (X^6)
)
d_in = input.shape[1]
d_out = 2 

#Calculate weight matrices
torch.manual_seed(123)  # Set a random seed for reproducibility
We_q = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)  # Query weight matrix
We_k = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)  # Key weight matrix
We_v = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)  # Value weight matrix
#Calculate Q, K, V for all tokens in the input sequence
queries, keys, values = compute_qkv(input, We_q, We_k, We_v)
print("Query vectors for all tokens:", queries)
print("Key vectors for all tokens:", keys)
print("Value vectors for all tokens:", values)


#Calculate attention scores for all tokens with respect to the query vector of the second token "journey"
attention_scores = calculate_attention_scores(queries[1], keys)  # Use the query vector of "journey" (second token) to calculate attention scores with respect to all key vectors
print("Attention scores for 'journey' with respect to all tokens:", attention_scores, "\n""\n")


#For all queries, calculate attention scores with respect to all keys
all_attention_scores = calculate_attention_scores(queries, keys)  # Compute the attention scores for all query vectors with respect to all key vectors
print("Attention scores for all queries with respect to all keys:\n", all_attention_scores)