import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
import string
import re
import nltk

from sklearn.datasets import fetch_20newsgroups

#KNN on 20NewsGroup

#Cabeçalho
st.title('KNN 20NewsGroup')

#Selecionar todas as categorias
user_all = st.checkbox('Todas as categorias')
if user_all == True:
    user_multi = st.multiselect("Selecione as categorias",options =['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc'], default = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc'] )
else:
    user_multi = st.multiselect("Selecione as categorias",options =['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc'] )

cats = user_multi

vizinhos = st.slider("Qtd vizinhos", min_value = 1, max_value = 50, value = 5)

a = st.button("Treinar")

#imprimindo se houver algo selecionado
if (len(cats) > 0) and (a):
     # Contagem de documentos de treino e teste por label
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories = cats )
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories = cats)

    X_train = np.array(newsgroups_train.data)
    y_train = np.array(newsgroups_train.target)
    X_test = np.array(newsgroups_test.data)
    y_test = np.array(newsgroups_test.target)    

    # Contagem de documentos de treino e teste por label


    def conta_labels(y_train, y_test):
        """Retorna dataframe com os total de documentos em cada classe
        de treinamento e teste. Ref.: Cachopo (2007)"""
        y_train_classes = pd.DataFrame([newsgroups_train.target_names[i] for i in newsgroups_train.target])[0]
        y_test_classes = pd.DataFrame([newsgroups_test.target_names[i] for i in newsgroups_test.target])[0]
        
        contagem_df = pd.concat([y_train_classes.value_counts(),
                                y_test_classes.value_counts()],
                                axis=1, 
                                keys=["# docs treino", "# docs teste"], 
                                sort=False)
        
        contagem_df["# total docs"] = contagem_df.sum(axis=1)
        contagem_df.loc["Total"] = contagem_df.sum(axis=0)
        
        return contagem_df

    newsgroups_df_labels = conta_labels(y_train, y_test)
    st.dataframe(newsgroups_df_labels)

    # Classe de Pre-Processamento de textos utilizando a biblioteca NLTK

    class NLTKTokenizer():
        """Classe que recebe documentos como entrada e devolve realizado lematização
        e retirando stopwords e pontuacoes.
        Ref.: https://scikit-learn.org/stable/modules/feature_extraction.html
        """    
        def __init__(self):
            self.lemmatizer = nltk.stem.WordNetLemmatizer()
            self.stopwords = nltk.corpus.stopwords.words('english')
            self.english_words = set(nltk.corpus.words.words())
            self.pontuacao = string.punctuation

        def __call__(self, doc):
            # ETAPA 1 - Limpeza de texto
            # Converte para minúsculo
            doc = doc.lower()       
            
            # Trocar numeros pela string numero
            doc = re.sub(r'[0-9]+', 'numero', doc)
            
            # Trocar underlines por underline
            doc = re.sub(r'[_]+', 'underline', doc)

            # Trocar URL pela string httpaddr
            doc = re.sub(r'(http|https)://[^\s]*', 'httpaddr', doc)
            
            # Trocar Emails pela string emailaddr
            doc = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', doc) 
            
            # Remover caracteres especiais
            doc = re.sub(r'\\r\\n', ' ', doc)
            doc = re.sub(r'\W', ' ', doc)

            # Remove caracteres simples de uma letra
            doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc)
            doc = re.sub(r'\^[a-zA-Z]\s+', ' ', doc) 

            # Substitui multiplos espaços por um unico espaço
            doc = re.sub(r'\s+', ' ', doc, flags=re.I)
            
            # ETAPA 2 - Tratamento da cada palavra
            palavras = []
            for word in nltk.word_tokenize(doc):
                if word in self.stopwords:
                    continue
                if word in self.pontuacao:
                    continue
                if word not in self.english_words:
                    continue
                
                word = self.lemmatizer.lemmatize(word)
                palavras.append(word)
            
            return palavras

    # Vetores de características com NLTK (lematização, remoção de stopwords e palavras desconhecidas)

    vetorizador_tratado = CountVectorizer(tokenizer=NLTKTokenizer())
    v2 = vetorizador_tratado.fit_transform(X_train)

    features = vetorizador_tratado.get_feature_names()
    v2_df = pd.DataFrame(v2.toarray(), columns = features)
    st.dataframe(v2_df)
    st.text(v2_df.shape)

    # KNN

    from sklearn.neighbors import KNeighborsClassifier

    text_clf_knn = Pipeline([('vect', CountVectorizer(tokenizer=NLTKTokenizer())),
                        ('tfidf', TfidfTransformer()),
                        ('clf', KNeighborsClassifier(n_neighbors=vizinhos, 
                                                    weights='uniform', 
                                                    algorithm='auto', 
                                                    leaf_size=30, 
                                                    p=2, 
                                                    metric='minkowski', 
                                                    metric_params=None, 
                                                    n_jobs=None)),
                        ])

    text_clf_knn.fit(X_train, y_train)
    predicted = text_clf_knn.predict(X_test)
    st.text(metrics.classification_report(y_test, predicted))


#aplicando o modelo

#predictor = ktrain.load_predictor('bertimbau9010procleve2')
#y = int(predictor.predict(user_input2))

#Resposta
#if (y == 1 and st.sidebar.button('Go')):
#    st.write(user_input,", você está assim: 😡!!! ")
#if (y == 2 and st.sidebar.button('Go')):
#    st.write(user_input,", você está assim: 🙁!!! ")
#if (y == 3 and st.sidebar.button('Go')):
#    st.write(user_input,", você está assim: 😐!!! ")
#if (y == 4 and st.sidebar.button('Go')):
#    st.write(user_input,", você está assim: 🙂!!! ")
#if (y == 5 and st.sidebar.button('Go')):
#    st.write(user_input,", você está assim: 😀!!! ")
