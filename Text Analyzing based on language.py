from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from stop_words import get_stop_words
from nltk.corpus import stopwords
import string
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud 
from nltk.stem import WordNetLemmatizer

def listToString(s):    
    str1 = ""  
    for ele in s:  
        str1 +=' '+ ele      
  
    return str1+'\n'


f= open("Text Analyzing based on language (Data).txt","r",encoding="ISO-8859-1")
f1=f.readlines()

Loweres_texts=[x.lower() for x in f1]
Loweres_texts=[''.join(x for x in par if x not in string.punctuation) for par in Loweres_texts]#remove punctuation from a list
Loweres_texts=[''.join(c for c in par if not c.isdigit()) for par in Loweres_texts]#remove digits from a list

word_tokens = sorted(set(word for sentence in Loweres_texts for word in sentence.split()))
word_tokens=list(filter(str.strip, word_tokens))#remove space in a list

stop_words_fr = get_stop_words('fr')
stop_words_sp = get_stop_words('spanish')
stop_words_gr = get_stop_words('german')
stop_words_en = stopwords.words('english')
stop_word=stop_words_fr+stop_words_sp+stop_words_gr+stop_words_en

Final=pd.DataFrame()
filtered_sentence = [w for w in word_tokens if not w in stop_word]  

vec = CountVectorizer(binary=False)#Convert a collection of text documents to a matrix of token counts
vec.fit(filtered_sentence)#Learn a vocabulary dictionary of all tokens in the raw documents.

Final=pd.DataFrame(vec.transform(Loweres_texts).toarray(), columns=sorted(vec.vocabulary_.keys()))

wordnet_lemmatizer = WordNetLemmatizer()
c=list(Final.columns)
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in c]
lemmatized_word=[listToString(lemmatized_word)]
dic = sorted(word for sentence in lemmatized_word for word in sentence.split())
Final.columns=dic

Final=Final.groupby(Final.columns, axis=1).sum() 

Language = pd.Series([]) 
for i in range(len(Final)): 
    if Final["english"][i] == 1: 
        Language[i]="English"
  
    elif Final["french"][i] == 1: 
        Language[i]="French"
  
    elif Final["spanish"][i] == 1: 
        Language[i]="Spanish"
        
    elif Final["german"][i] == 1: 
        Language[i]="German"
    else:
            Language[i]= Final["german"][i] 
    
Final.insert(0, "Language",Language, True) 
Final=Final.drop(columns=['english','french','spanish','german'])
Final_2=Final.groupby("Language").apply(np.sum)
Final_2=Final_2.drop(columns=['Language'])

wordcloud = WordCloud(width=1000, height=1000, contour_color='steelblue').generate(Final_2.to_string()) 

# plot the WordCloud image  
plt.figure(figsize=(60,60)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.margins(x=0, y=0) 
plt.show()
