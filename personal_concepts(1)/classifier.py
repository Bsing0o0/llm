# Store a Model ( Dictionary of words -> weight)
#Train a tiny dataset
#Predict the class of new text

#(OOP, dict, Lambda, methods)

class MiniTextClassifier:
    def __init__(self, *words, **numb):
        self.word = words
        self.frequency = numb
        
    def dictionary(self):
        #Build a dictionary of words and their frequencies
        Dict = lambda: {word: self.frequency.get(word, 0) for word in self.word}
        print (Dict(), "This is result from ", self.frequency, self. word)
        

Dictionary = MiniTextClassifier("hello", "world", "gimme", hello=5, world=3)
Dictionary.dictionary()