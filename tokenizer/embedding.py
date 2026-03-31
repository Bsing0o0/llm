import torch
from input_target import dataloaderv1  #  DataLoader function




# Parameters according to CHTGPT - 2 model
vocab_size = 50356       # size of embedding matrix 
embedding_dim = 300      # dimension of each token embedding
context_size = 4         # number of tokens per input sequence
batch_size = 8           # mini-batch size



# Create Embedding Layer
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)


# Create DataLoader from raw text
file_path = "C:/Users/balka/OneDrive/Documents/Personal Projects/llm_tokenizer/data/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = dataloaderv1(
    raw_text=raw_text,
    batch_size=batch_size,
    context_size=context_size,
    stride=1,       # overlapping sequences
    shuffle=False
)


for input_batch, output_batch in dataloader:
    
    # Convert token IDs → embedding vectors
    embedded_matrix = token_embedding_layer(input_batch)  # shape: [batch_size, context_size, embedding_dim]
    
    
    
    
    
    print (embedded_matrix)