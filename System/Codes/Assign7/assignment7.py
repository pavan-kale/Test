text= "My name is vishnu. I am from solapur. I am pursuing my degree from Shreemati kashibai Navale College of Engineering, vadgaon pune 411310."
import re
import nltk
#Sentence Tokenization 
from nltk.tokenize import sent_tokenize 
tokenized_text= sent_tokenize(text) 
print(tokenized_text)
#Word Tokenization 
from nltk.tokenize import word_tokenize 
tokenized_word=word_tokenize(text) 
print(tokenized_word)
# Print stop words of English 
from nltk.corpus import stopwords 
stop_words=set(stopwords.words("english")) 
print(stop_words)
text= "How to remove stop words with NLTK library in Python?" 
text= re.sub('[^a-zA-Z]', ' ',text) 
tokens = word_tokenize(text.lower()) 
filtered_text=[] 
for w in tokens: 
	if w not in stop_words: 
		filtered_text.append(w) 
print ("Tokenized Sentence:",tokens)
print ("Filterd Sentence:",filtered_text)
from nltk.stem import PorterStemmer 
e_words= [ "Wait", "waiting", "waited", "waits"] 
ps =PorterStemmer() 
for w in e_words: 
	rootWord=ps.stem(w) 
	print(" ",rootWord)
from nltk.stem import WordNetLemmatizer 
wordnet_lemmatizer = WordNetLemmatizer() 
text = "studies studying cries cry" 
tokenization = nltk.word_tokenize(text) 
for w in tokenization: 
	print("Lemma for {} is {}".format(w,wordnet_lemmatizer.lemmatize(w)))
from nltk.tokenize import word_tokenize 
data="The pink sweater fit her perfectly" 
words=word_tokenize(data) 
for word in words: 
	print(nltk.pos_tag([word]))