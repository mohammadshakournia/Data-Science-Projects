from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import en_core_web_sm
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #
import numpy as np

#function for convert a list to a string 
def listToString(s):    
    str1 = ""  
    for ele in s:  
        str1 +=' '+ ele      
    return str1+'\n'

#function for return first charecters of a string
def first_char(s,f,n):
    return s[f:n]
#create a pandas dataframe for store results
Final=pd.DataFrame();
#open the source dataset file
f= open("SemEval2017-task4-dev.subtask-A.english.INPUT.txt","r",encoding="ISO-8859-1")
f1=f.readlines()
#delete elements of f1 list for easy running
del f1[15000:20632]
#take 18 first charecture as id number and store then into a series
id_num=[first_char(w,0,18) for w in f1]
id_numbers = pd.Series(id_num) 
#take 8 charecters from index 19 to 27 as status (negetive,positive,neutral) and store then into a series
status=[first_char(w,19,27) for w in f1]
status=[w.replace('\t','') for w in status]#remove space
Status = pd.Series(status)
#take other charecters as main sentances
f1=[s[27:] for s in f1]
#word tokenizing to get list of word in whole documnet
word_token=list(set(word for sentence in f1 for word in sentence.split()))
#remove numbers
word_token=[''.join(c for c in par if not c.isdigit())for par in word_token]
#store punctuation in a list except '@' and '#' because some of word which have this punctuation maybe are a name_entity
punctuation=[x for x in string.punctuation]
for w in ['#','@']:
      punctuation.remove(w)
#remove punctuation
word_token=[''.join(x for x in par if x not in punctuation) for par in word_token]
word_token=list([x for x in word_token if x !=''])#remove space
#convert list of words to a string because nlp function for detecting name_entities just take string as input
word_token_text=listToString(word_token)
#detecting name_entites using en_core_web_sm and store them in a list
nlp=en_core_web_sm.load()
text=nlp(word_token_text)
name_entity=list(x.text for x in text.ents)
name_entity=list(word for sentence in name_entity for word in sentence.split())
#remove name_entities from list of main words which we got from main text
word_token_ne_removed=[x for x in word_token if x not in name_entity]
#remove '@' from word_token_ne_removed list because we don't need to them anymore
word_token_ne_removed=[w for w in word_token_ne_removed if not first_char(w,0,1)=='@']
#convert all of word to lowercase
Loweres_texts=[x.lower() for x in word_token_ne_removed]
#store stop word to a list
stop_words_en = stopwords.words('english')
#some of stop words like 'dont`t' maybe change our sentence meaning and also change their status so we don't remove them 
for w in ['not','no','don\'t','aren\'t','couldn\'t','didn\'t','doesn\'t','hasn\'t','haven\'t'
          ,'isn\'t','mightn\'t','mustn\'t','needn\'t','shan\'t','shouldn\'t','wasn\'t','weren\'t'
          ,'won\'t','wouldn\'t']:
    stop_words_en.remove(w)
#remove stop words
filtered_sentence =[w for w in Loweres_texts if not w in stop_words_en]
#remove hyperlinks 
filtered_sentence=[w for w in filtered_sentence if not first_char(w,0,4)=='http']
#stemming on words
ps =PorterStemmer()
filtered_sentence =[ps.stem(word) for word in filtered_sentence]
#lemmatizing on words
wordnet_lemmatizer = WordNetLemmatizer()
filtered_sentence =[wordnet_lemmatizer.lemmatize(word) for word in filtered_sentence]
filtered_sentence =[w for w in filtered_sentence if not w in stop_words_en]
#because stemming make some words in wrong spell so we use spellcheckr to make them correct but for i don't run it here beacause of problem in compiling time
#spell = SpellChecker()
#filtered_sentence =sorted([spell.correction(word) for word in filtered_sentence])

#Convert a collection of text documents to a matrix of token counts
vec = CountVectorizer(binary=False)
#Learn a vocabulary dictionary of all tokens in the raw documents
vec.fit(filtered_sentence)
#store results data into dataframe 
Final=pd.DataFrame(vec.transform(f1).toarray(), columns=sorted(vec.vocabulary_.keys()))
#using TF-IDF to show calculate importance of a word 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(Final)
idf = vectorizer.idf_
# make a new dictionary of word after calculate TF-IDF and update dataframe 
dict1=dict(zip(vectorizer.get_feature_names(), idf))
Final=pd.DataFrame(vectorizer.transform(f1).toarray(),columns=dict1.keys())
#to do a feature selection at the first I calculate sum of the TF-IDF value of each features
#then import summation and name of feature in a new dataframe
summation=list(Final.sum(axis = 0, skipna = True))
column=list(Final.columns)
sum_col=pd.DataFrame(columns=['sum','Col'])
sum_col['Col']=column
sum_col['sum']=summation
#after calculate the summation ,i removed zero values from dataframe
l=[]
for e in sum_col.values:
    if e[0]==0: # if value of summation be zero then append its feature name to list
        l.append(e[1])
Final=Final.drop(columns=l) #remove the feature in list from dataframe

#calculate summation again but this time i remove the values less than mean of TF-IDF in whole feature space
summation=list(Final.sum(axis = 0, skipna = True))
mean=(sum_col['sum'].sum(axis = 0, skipna = True))/(len(Final.columns))

#update sum_col dataframe
column=list(Final.columns)
sum_col=pd.DataFrame(columns=['sum','Col'])
sum_col['Col']=column
sum_col['sum']=summation

l=[]
for e in sum_col.values:
    if e[0]<=mean:# if value of summation be less than mean then append its feature name to list
        l.append(e[1])
Final=Final.drop(columns=l)
#add id number and status of each comment that we split them at first to result dataframe
#Final['ID_Number']=id_numbers.values
Final['Status']=Status.values
cols = Final.columns.tolist()
#cols.insert(0, cols.pop(cols.index('ID_Number')))
cols.insert(0, cols.pop(cols.index('Status')))
Final = Final.reindex(columns= cols)

Final['Status']=Final['Status'].map({'negative':0,'neutral':1,'positive':2})

feature_cols = cols
feature_cols.remove('Status')
X = Final[feature_cols] # Features
y = Final.Status # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#custom_tweet =["Thank you to Brenda May Callander jp Peter tanya I think briana  zack Austin Andrew  and I'm thinking grandma"]
#custom_tweet=[x.lower() for x in custom_tweet]
#test1=list(set(word for sentence in custom_tweet for word in sentence.split()))
#
#vec = CountVectorizer(binary=False)
##Learn a vocabulary dictionary of all tokens in the raw documents
#vec.fit(test1)
##store results data into dataframe 
#df1=pd.DataFrame(vec.transform(custom_tweet).toarray(), columns=sorted(vec.vocabulary_.keys()))
#result = pd.concat([df1, Final], axis=1, sort=True, join='inner')
#result_vcol=list(set(result.columns))
#n=['john','josh','jason','ireland','lesnar','motorola','muslim','rock','summerslam','tanya','trump','twitter','ufc','uk','zack','Status','al']
#result_vcol=list(set(result_vcol)-set(n))
#result1=pd.DataFrame(0,index=range(1),columns=result_vcol)
#result = pd.concat([result1, df1],sort=True)
#result.fillna(0, inplace=True)
#pre = clf.predict(result)
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False)
