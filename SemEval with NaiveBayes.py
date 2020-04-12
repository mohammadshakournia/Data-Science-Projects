import pandas as pd
import re, string, random
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import  stopwords
from nltk import FreqDist, classify, NaiveBayesClassifier

def first_char(s,f,n):
    return s[f:n]

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

f= open("SemEval2017-task4-dev.subtask-A.english.INPUT.txt","r",encoding="ISO-8859-1")
Input_Data=f.readlines()
del Input_Data[18000:20632]

status=[first_char(w,19,27) for w in Input_Data]
status=[w.replace('\t','') for w in status]
Status = pd.Series(status)

Input_Data=[s[27:] for s in Input_Data]
Input_Data=[w.replace('\t','') for w in Input_Data]
data = pd.Series(Input_Data)

df=pd.DataFrame(columns={'s','d'})
df['d']=data.values
df['s']=Status.values

negative_comments=df.loc[df['s'] == 'negative']
negative_comments=list(negative_comments['d'].values)
positive_comments=df.loc[df['s'] == 'positive']
positive_comments=list(positive_comments['d'].values)
neutral_comments=df.loc[df['s'] == 'neutral']
neutral_comments=list(neutral_comments['d'].values)

del status,Status,data,df
#sample=list(set(word for sentence in Input_Data for word in sentence.split()))
negative_comment_tokens=[word_tokenize(w) for w in negative_comments]
neutral_comment_tokens=[word_tokenize(w) for w in neutral_comments]
positive_comment_tokens=[word_tokenize(w) for w in positive_comments]

stop_words = stopwords.words('english')

negetive_clean_token=[]
neutral_clean_token=[]
positive_clean_token=[]

for token in negative_comment_tokens:
    negetive_clean_token.append(remove_noise(token,stop_words))
for token in neutral_comment_tokens:
    neutral_clean_token.append(remove_noise(token,stop_words))
for token in positive_comment_tokens:
    positive_clean_token.append(remove_noise(token,stop_words))
    
all_pos_words = get_all_words(positive_clean_token)
all_neg_words = get_all_words(negetive_clean_token)
freq_dist_pos = FreqDist(all_pos_words)
freq_dist_neg = FreqDist(all_neg_words)

print(freq_dist_pos.most_common(10))
print(freq_dist_neg.most_common(10))

negative_token_for_model=get_tweets_for_model(negetive_clean_token)
neutral_token_for_model=get_tweets_for_model(neutral_clean_token)
positive_token_for_model=get_tweets_for_model(positive_clean_token)

negative_dataset=[(tokens,'negative') for tokens in negative_token_for_model]
neutral_dataset=[(tokens,'neutral') for tokens in neutral_token_for_model]
positive_dataset=[(tokens,'positive') for tokens in positive_token_for_model]

dataset=negative_dataset+neutral_dataset+positive_dataset
random.shuffle(dataset)

train_data = dataset[:12000]
test_data = dataset[12000:]

classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))

custom_tweet = "Thank you to Brenda May Callander jp Peter tanya (I think) briana  zack Austin Andrew  and I'm thinking grandma... https://t.co/44O5xvolkn"

custom_tokens = remove_noise(word_tokenize(custom_tweet))

print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))

