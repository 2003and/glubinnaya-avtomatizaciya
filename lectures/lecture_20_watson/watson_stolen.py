import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

## Data Visualisation
from plotly.offline import iplot
from plotly import tools
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import plotly.offline as pyo

## Data Preprocessing
import re
import nltk
from gensim.models import word2vec

## Visializing similarity of words
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

##Translation
#from googletrans import Translator


## Models
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
from sklearn.preprocessing import LabelEncoder


train_df = pd.read_csv("/home/andrew/VS/glubinnaya-avtomatizaciya/lectures/lecture_20_watson/train.csv")
test_df = pd.read_csv("/home/andrew/VS/glubinnaya-avtomatizaciya/lectures/lecture_20_watson/test.csv")

Accuracy=pd.DataFrame()
Accuracy['Type']=train_df.label.value_counts().index
Accuracy['Count']=train_df.label.value_counts().values
Accuracy['Type']=Accuracy['Type'].replace(0,'Entailment')
Accuracy['Type']=Accuracy['Type'].replace(1,'Neutral')
Accuracy['Type']=Accuracy['Type'].replace(2,'Contradiction')

Languages=pd.DataFrame()
Languages['Type']=train_df.language.value_counts().index
Languages['Count']=train_df.language.value_counts().values

Languages_test=pd.DataFrame()
Languages_test['Type']=test_df.language.value_counts().index
Languages_test['Count']=test_df.language.value_counts().values
a = sum(Languages_test.Count)
Languages_test.Count = Languages_test.Count.div(a).mul(100).round(2)

a = sum(Languages.Count)
Languages.Count = Languages.Count.div(a).mul(100).round(2)

Meta_features = pd.DataFrame()

## Number of words in the text ##
Meta_features["premise_num_words"] = train_df["premise"].apply(lambda x: len(str(x).split()))
Meta_features["hypothesis_num_words"] = train_df["hypothesis"].apply(lambda x: len(str(x).split()))

## Number of characters in the text ##
Meta_features["premise_num_chars"] = train_df["premise"].apply(lambda x: len(str(x)))
Meta_features["hypothesis_num_chars"] = train_df["hypothesis"].apply(lambda x: len(str(x)))

## Number of punctuations in the text ##
Meta_features["premise_num_punctuations"] =train_df["premise"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
Meta_features["hypothesis_num_punctuations"] =train_df["hypothesis"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Average length of the words in the text ##
Meta_features["premise_mean_word_len"] = train_df["premise"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
Meta_features["hypothesis_mean_word_len"] = train_df["hypothesis"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

Meta_features['label'] = train_df['label']



temp = pd.DataFrame()
temp['premise'] = train_df['premise']
temp['hypothesis'] = train_df['hypothesis']

STOP_WORDS = nltk.corpus.stopwords.words()

def clean_sentence(val):
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")
    
    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)  
            
    sentence = " ".join(sentence)
    return sentence

temp['premise'] =  temp['premise'].apply(clean_sentence)
temp['hypothesis'] =  temp['hypothesis'].apply(clean_sentence)

def build_corpus(data):
    corpus = []
    for col in ['premise', 'hypothesis']:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
            
    return corpus

corpus = build_corpus(temp)        

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

# A more selective model
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=150, workers=4)
tsne_plot(model)

## Number of words in the text ##
train_df["premise_num_words"] = train_df["premise"].apply(lambda x: len(str(x).split()))
train_df["hypothesis_num_words"] = train_df["hypothesis"].apply(lambda x: len(str(x).split()))
test_df["premise_num_words"] = test_df["premise"].apply(lambda x: len(str(x).split()))
test_df["hypothesis_num_words"] = test_df["hypothesis"].apply(lambda x: len(str(x).split()))

## Number of characters in the text ##
train_df["premise_num_chars"] = train_df["premise"].apply(lambda x: len(str(x)))
train_df["hypothesis_num_chars"] = train_df["hypothesis"].apply(lambda x: len(str(x)))
test_df["premise_num_chars"] = test_df["premise"].apply(lambda x: len(str(x)))
test_df["hypothesis_num_chars"] = test_df["hypothesis"].apply(lambda x: len(str(x)))

## Number of punctuations in the text ##
train_df["premise_num_punctuations"] =train_df["premise"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
train_df["hypothesis_num_punctuations"] =train_df["hypothesis"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test_df["premise_num_punctuations"] = test_df["premise"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test_df["hypothesis_num_punctuations"] = test_df["hypothesis"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Average length of the words in the text ##
train_df["premise_mean_word_len"] = train_df["premise"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
train_df["hypothesis_mean_word_len"] = train_df["hypothesis"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_df["premise_mean_word_len"] = test_df["premise"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_df["hypothesis_mean_word_len"] = test_df["hypothesis"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

## Language Transformation
lb_make = LabelEncoder()
train_df["language"] = lb_make.fit_transform(train_df["language"])
test_df["language"] = lb_make.fit_transform(test_df["language"])
                                         
## lang_abv Transformation
lb_make = LabelEncoder()
train_df["lang_abv"] = lb_make.fit_transform(train_df["lang_abv"])
test_df["lang_abv"] = lb_make.fit_transform(test_df["lang_abv"])

from nltk.corpus import stopwords
import re
import nltk
import string

stop_words = set(stopwords.words('english')) 
def text_cleaner(text):
    newString = text.lower()
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    tokens = [w for w in newString.split() if not w in stop_words]
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()

cleaned_text = []
for t in train_df['premise']:
    cleaned_text.append(text_cleaner(t))
train_df['premise'] = cleaned_text   

cleaned_text = []
for t in test_df['premise']:
    cleaned_text.append(text_cleaner(t))
test_df['premise'] = cleaned_text 

cleaned_text = []
for t in train_df['hypothesis']:
    cleaned_text.append(text_cleaner(t))
train_df['hypothesis'] = cleaned_text   

cleaned_text = []
for t in test_df['hypothesis']:
    cleaned_text.append(text_cleaner(t))
test_df['hypothesis'] = cleaned_text 

## premise
tfidf_vec = TfidfVectorizer(analyzer='word',max_features=1000)
tfidf_vec.fit(train_df['premise'].values.tolist() + test_df['premise'].values.tolist())
train_premise = tfidf_vec.transform(train_df['premise'].tolist())
df1 = pd.DataFrame(train_premise.toarray(), columns=tfidf_vec.get_feature_names()).add_suffix('_premise')
train_df = pd.concat([train_df, df1], axis = 1)

test_premise = tfidf_vec.transform(test_df['premise'].tolist())
df1 = pd.DataFrame(test_premise.toarray(), columns=tfidf_vec.get_feature_names()).add_suffix('_premise')
test_df = pd.concat([test_df, df1], axis = 1)

## premise
tfidf_vec = TfidfVectorizer(analyzer='word',max_features=1000)
tfidf_vec.fit(train_df['hypothesis'].values.tolist() + test_df['hypothesis'].values.tolist())
train_premise = tfidf_vec.transform(train_df['hypothesis'].tolist())
df1 = pd.DataFrame(train_premise.toarray(), columns=tfidf_vec.get_feature_names()).add_suffix('_hypothesis')
train_df = pd.concat([train_df, df1], axis = 1)

test_premise = tfidf_vec.transform(test_df['hypothesis'].tolist())
df1 = pd.DataFrame(test_premise.toarray(), columns=tfidf_vec.get_feature_names()).add_suffix('_hypothesis')
test_df = pd.concat([test_df, df1], axis = 1)

def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, child=1, colsample=0.3):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 3
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = child
    param['subsample'] = 0.8
    param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    num_rounds = 2000

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
    return pred_test_y, pred_test_y2, model

train_X = train_df.drop(list(train_df.columns[[0,1]])+['label']+['premise','hypothesis'], axis=1)
test_X = test_df.drop(list(test_df.columns[[0,1]])+['premise','hypothesis'], axis=1)
train_y = train_df['label']

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], 3])

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0, colsample=0.7)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    break
print("cv scores : ", cv_scores)

out_df = pd.DataFrame(pred_full_test)

submission = pd.DataFrame()
submission['id'] = test_df['id']
submission['prediction'] = out_df.idxmax(axis=1)

submission.to_csv("submission.csv", index=False)

