import torch
import tiktoken

from torch.utils.data import Dataset, DataLoader

class datasetv1(Dataset):
    def __init__(self, raw_text, tokenizer, context_size, stride):
        self.input_ids = []
        self.output_ids = []
        
        #tokenize
        token_ids = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
        
        # Sliding window 
        for i in range (0, len(token_ids) - context_size, stride):
            input_chunk = token_ids[i:i+context_size]
            output_chunk = token_ids[i+1: i+context_size +1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.output_ids.append(torch.tensor(output_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_ids[idx]
    
    
    
def dataloaderv1( raw_text, batch_size = 4, context_size = 256, stride = 128, shuffle = True, drop_last = True, num_workers = 0):
    
    # inisialize BPE tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    #Create Dataset
    
    dataset = datasetv1(raw_text, tokenizer, context_size, stride)
    
    #Create DataLoader
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last, num_workers = num_workers)
    return dataloader


#Testing the DataLaoder
raw_text = "Testing the Dataloader and dataset created using this text that tokenize using tiktoken BPE tokenizer. This is a simple test to see if the dataloader and dataset are working correctly."

loader = dataloaderv1(raw_text, batch_size = 1, context_size = 4, stride = 1)

for iteration_idx, (input, output) in enumerate(loader):
    print(f"batch {iteration_idx}:")
    print("Input IDs:", input)
    print("Output IDs:", output)
    
#testing with new text
with open("C:/Users/balka/OneDrive/Documents/Personal Projects/llm_tokenizer/data/the-verdict.txt", "r", encoding="utf-8") as f:
    new_text = f.read()
new_loader = dataloaderv1(new_text)

for iteration_idx2, (input2, output2) in enumerate(new_loader):
    print(f"batch {iteration_idx2}:")
    print("Input IDs:", input2)
    print("Output IDs:", output2)