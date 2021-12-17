#!/usr/bin/env python3
import streamlit as st

import pandas as pd
import numpy as np
import gzip
import gensim.downloader as api
from gensim import corpora
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
import os
import itertools
from collections import Counter
import re
import string
from textblob import Word
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from spellchecker import SpellChecker
@st.cache
def importish():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
importish()
tknzr = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
spell=SpellChecker()
# Creating our tokenizer and lemmatizer
@st.cache
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F" # emoticons
                           u"\U0001F300-\U0001F5FF" # symbols & pictographs
                           u"\U0001F680-\U0001F6FF" # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642" 
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)
@st.cache
def clean_text(text):
    # remove numbers
    text=remove_emoji(text)
    text_nonum = re.sub(r'\d+', '', text)
    text_no_misplacedstop = text_nonum.replace('.',' ')
    text_no_forslash = text_no_misplacedstop.replace('/',' ')
    answer = tknzr.tokenize(text_no_forslash)
    SpellChecked=[spell.correction(word) for word in answer]
    answer =' '.join(SpellChecked)
    # remove punctuations and convert characters to lower case
    punc = string.punctuation.replace('.','')
    punc = punc.replace('/','')
    text_no_punc = "".join([char.lower() for char in answer if char not in punc]) 
    # substitute multiple whitespace with single whitespace
    # removes leading and trailing whitespaces
    # remove forward slash with space
    text_no_doublespace = re.sub('\s+', ' ',text_no_punc).strip()
    return text_no_doublespace
stpwrd = nltk.corpus.stopwords.words('english')
stpwrd.extend(string.punctuation)
keepwords="don't,does,no,not,can,should,will,aren't,couldn't,doesn't,isn't,shouldn't,won't,is".split(',')
for word in keepwords:
    stpwrd.remove(word)
def lem_data(data):
  tknzr = TweetTokenizer()  
  data = tknzr.tokenize(data)   
  data = [word for word in data if word not in stpwrd]  
  data = [lemmatizer.lemmatize(x) for x in data]
  return data
@st.cache
def questiontype(df):
  YeeeNooo=df[df['question'].str.contains('does|can|will|would',flags=re.IGNORECASE)].index.to_list() 
  df['qtype']='open-ended'
  df.at[YeeeNooo,'qtype']='yes/no'
  return df
@st.cache
def yes_no(df):
# yes/no helpful replies
    Yes_No=df[df['answer'].str.contains('definitely|absolutely|positively|suppose so|believe so|think so',flags=re.IGNORECASE,regex=True)].index.to_list()
    Yes=df[df['answer'].str.contains('yes',flags=re.IGNORECASE,regex=False)].index.to_list()
    No=df[df['answer'].str.contains('no',flags=re.IGNORECASE,regex=False)].index.to_list()
    Not=df[df['answer'].str.contains('not',flags=re.IGNORECASE,regex=False)].index.to_list()
    df.at[Yes_No+Yes+No+Not,'Helpful-Definitive']=1
    
    definitively_definitive=df[((df['answerType']=='Y')|(df['answerType']=='N'))&(df['Helpful-Definitive']==0)].index.to_list()
    for x in definitively_definitive:
        df.at[x,'Helpful-Definitive']=1 
    return df
@st.cache
def specboyQ(df): 
        # definitively unhelpful replies
    idk=df[df['answer'].str.contains("don't know|not sure|do not know|can't help|not arrived|gift",flags=re.IGNORECASE)].index.to_list()
    df['Unhelpful']=0
    df.at[idk,'Unhelpful']=1
    size=df[(df['answer'].str.contains('"|width|height|wide|long|tall|high|inch|measures|inch|metre|meter|feet|cm|centimetre|millimetre|big|small|large|tiny',flags=re.IGNORECASE))&(df['question'].str.contains('size|big|high|wide|diameter|clearance|clearence|dimension|dimention|depth|height|width|high|wide|measure',flags=re.IGNORECASE))].index.to_list()
    where=df[(df['answer'].str.contains('under|behind|top|bottom|left|right|side|front|back|over|below|inside|outside',flags=re.IGNORECASE))&(df['question'].str.contains('where',flags=re.IGNORECASE))].index.to_list()
    itis=df[(df['answer'].str.contains("its|it's|it is",flags=re.IGNORECASE))&(df['question'].str.contains('is this|is it',flags=re.IGNORECASE))].index.to_list()
    how=df[(df['answer'].str.contains("use|using|have to",flags=re.IGNORECASE))&(df['question'].str.contains('how',flags=re.IGNORECASE))].index.to_list()
    can=df[(df['answer'].str.contains("can",flags=re.IGNORECASE))&(df['question'].str.contains('can',flags=re.IGNORECASE))].index.to_list()
    inclusive=df[(df['answer'].str.contains('came with|comes with|include',flags=re.IGNORECASE))&(df['question'].str.contains('come with|include',flags=re.IGNORECASE))].index.to_list()
    QAnswered=inclusive+where+itis+size+how+can
    df['Helpful-QAnswered']=0
    df.at[QAnswered,'Helpful-QAnswered']=1
    return df
w2v_model = api.load("glove-wiki-gigaword-50")
similarity_index = WordEmbeddingSimilarityIndex(w2v_model)
@st.cache
def SCM(q, a): 
  """Function that calculates Soft Cosine Similarity between a Question and its Answer
     references: https://devopedia.org/question-similarity
    https://notebook.community/gojomo/gensim/docs/notebooks/soft_cosine_tutorial
    """

  q_lem = lem_data(q)
  a_lem = lem_data(a)
  documents = [q_lem,a_lem]
  dictionary = corpora.Dictionary(documents)

  # Convert the sentences into bag-of-words vectors.
  q_bag = dictionary.doc2bow(q_lem)
  a_bag = dictionary.doc2bow(a_lem)

  # Prepare the similarity matrix
  similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary)

  # compute SCM using the inner_product method
  similarity = similarity_matrix.inner_product(q_bag, a_bag, normalized=(True, True))

  # convert SCM score to percentage
  percentage_similarity= round(similarity * 100,2)
#f'\nThe percentage chance the answer is useful is {percentage_similarity}% similar.'
  return f'The answer is {percentage_similarity}% similar to the question.'
@st.cache
def is_useful(q, a, questionType, answerType):
    """Function that evaluates the usefulness of the answer to a question 
    in the Amazon reviews section"""
    
    d={'question':[q],'answer':[a],'questionType':[questionType],'answerType':[answerType]}
    df= pd.DataFrame(data=d)
    df['question'] = df['question'].apply(clean_text)
    df['answer'] = df['answer'].apply(clean_text)
    df=questiontype(df)
    df['Helpful-Definitive']=0
    df['Unhelpful']=0
    df=specboyQ(df)
    if (df['questionType'].iloc[0]=='yes/no')|(df['qtype'][0]=='yes/no'):
        df=yes_no(df)
    if len(df)==df['Unhelpful'].sum():
        return 'This answer is unhelpful'
    elif len(df)==df['Helpful-QAnswered'].sum()+df['Helpful-Definitive'].sum():
        return 'This answer is helpful'
    return SCM(q=df['question'].iloc[0],a=df['answer'].iloc[0])

st.title("How useful is the answer?")
st.text("Fire away!")
q = st.text_input("Ask a question", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None)
qtype = st.radio("What type of question is this?",["yes/no","open-ended"])
a = st.text_input("Answer the question", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None)
atype = st.radio("What type of answer is this?",["Y","N","Other","That is confidential"])
if st.button("Am I useful? 🥺", key=None, help=None, on_click=None, args=None, kwargs=None):
  answer=is_useful(q,a,qtype,atype)
  st.text({answer})


