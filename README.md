# llm
Learning llm step by step


Step 1 Tokenizer

I built a small Python tokenizer and encoder. It reads a text file, splits it into tokens, assigns unique IDs to each token, and can turn those IDs back into text. This was my way of exploring how language models process text.

Functions 
Splits text into words and punctuation
Creates a vocabulary of unique tokens
Converts tokens to numeric IDs
Converts IDs back into text
Can handle any text file


###How to Use
from tokenizer import Tokenizer

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = Tokenizer()
tokens = tokenizer.tokenize(text)
encoded_ids = tokenizer.encode(tokens)
decoded_text = tokenizer.decode(encoded_ids)

print("Tokens (first 10):", tokens[:10])
print("Encoded IDs (first 10):", encoded_ids[:10])
print("Decoded Text (first 100):", decoded_text[:100])
