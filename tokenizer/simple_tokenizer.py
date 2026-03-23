import re  # Import the Regular Expressions module for text splitting

# Define a simple tokenizer class
class SimpleTokenizer:
    def __init__(self):
        # Dictionaries to map tokens to IDs and back
        self.token_to_id = {}
        self.id_to_token = {}
        self.next_id = 0

    def tokenize(self, text):
        # Split text into words and punctuation using regex
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    def build_vocab(self, texts):
        # Build vocabulary from a list of texts
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.token_to_id:
                    # Assign a unique ID to each new token
                    self.token_to_id[token] = self.next_id
                    self.id_to_token[self.next_id] = token
                    self.next_id += 1

    def encode(self, text):
        # Convert tokens into numeric IDs
        tokens = self.tokenize(text)
        return [self.token_to_id.get(token, -1) for token in tokens]  # -1 for unknown tokens

    def decode(self, ids):
        # Convert IDs back to text
        return " ".join([self.id_to_token.get(i, "[UNK]") for i in ids])  # [UNK] for unknown IDs


if __name__ == "__main__":
    # Read your verdict text file
    with open("../data/the-verdict.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Create tokenizer instance
    tokenizer = SimpleTokenizer()

    # Build vocabulary from the text
    tokenizer.build_vocab([text])

    # Encode text to numeric IDs
    encoded = tokenizer.encode(text)

    # Decode IDs back to text
    decoded = tokenizer.decode(encoded)

    # Print outputs (first few chars/tokens for readability)
    print("Original Text:\n", text[:500], "...")
    print("\nTokens (first 100):\n", tokenizer.tokenize(text)[:100])
    print("\nEncoded IDs (first 100):\n", encoded[:100])
    print("\nDecoded Text (first 500):\n", decoded[:500], "...")