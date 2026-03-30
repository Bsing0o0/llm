#using google news dataset API

import gensim.downloader as api
model = api.load("word2vec-google-news-300")

word_vector = model

print (word_vector["king"])