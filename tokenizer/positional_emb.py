import torch
from input_target import dataloaderv1  #  DataLoader function

vocab_size = 50257      # size of embedding matrix 
embedding_dim = 256      # dimension of each token embedding

token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

context_size = 4         # number of tokens per input sequence
with open("C:/Users/balka/OneDrive/Documents/Personal Projects/llm_tokenizer/data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = dataloaderv1(
    raw_text=raw_text,
    batch_size=8,
    context_size= context_size,
    stride=context_size,       # overlapping sequences
    shuffle=False
)


data_iter = iter(dataloader)
input_batch, output_batch = next(data_iter)

###  print(input_batch)     # (8*4) tensor of token IDs

embedded_matrix = token_embedding_layer(input_batch)  # shape: [batch_size, context_size, embedding_dim]


###  print(embedded_matrix.shape)  # should print: torch.Size([8, 4, 256])


#Now we will make positional embedding to add it to the token embeddings
# Create Positional Embedding Layer

context_length = context_size  # number of tokens in the input sequence
positional_embedding_layer = torch.nn.Embedding(context_length, embedding_dim) # context_length was used because we want to encode the position of each token in the sequence, and embedding_dim is the same as token embedding dimension to allow addition.

# We only need to create positional embedding once, since the positions are fixed for a given context size. We can create a tensor of positions and pass it through the positional embedding layer.
positions = positional_embedding_layer(torch.arange(context_length))  # shape: [context_length, embedding_dim]
print(positions.shape)  # should print: torch.Size([4, 256])

 