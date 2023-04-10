import json
from json.tool import main
import jsonlines
import numpy as np
import math
import os
import multiprocessing
import argparse
import statistics
import codecs
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import operator
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="sciatica",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state='collapsed'
)


def dot_product2(v1, v2):
    return sum(map(operator.mul, v1, v2))


def cosine_sim(v1, v2):
    prod = dot_product2(v1, v2)
    len1 = math.sqrt(dot_product2(v1, v1))
    len2 = math.sqrt(dot_product2(v2, v2))
    return prod / (len1 * len2)

model_list = [['alberta', 2.5], ['all_mpnet_base_v2', 5], ['bert_nli', 0.5], ['bert_pp', 0.5], ['distilbert_nli', 5], ['allenai_specter', 3.5]]


@st.cache_resource
def load_model():
    model_sent_bert_nli = SentenceTransformer('nli-roberta-base-v2')

    model_sent_bert_pp = SentenceTransformer('paraphrase-TinyBERT-L6-v2')

    model_all_mpnet_base_v2 = SentenceTransformer('all-mpnet-base-v2')

    model_sent_distbert_nli = SentenceTransformer('all-distilroberta-v1')

    model_alberta = SentenceTransformer('paraphrase-albert-small-v2')

    specter_tokenize = AutoTokenizer.from_pretrained('allenai/specter')
    specter_model = AutoModel.from_pretrained('allenai/specter') 

    return model_sent_bert_nli, model_sent_bert_pp, model_all_mpnet_base_v2, model_sent_distbert_nli, model_alberta, specter_tokenize, specter_model

model_sent_bert_nli, model_sent_bert_pp, model_all_mpnet_base_v2, model_sent_distbert_nli, model_alberta, specter_tokenize, specter_model =  load_model()

def get_bert_nli_embedding(sentence):
    return model_sent_bert_nli.encode(sentence)

def get_bert_pp_embedding(sentence):
    return model_sent_bert_pp.encode(sentence)

def get_all_mpnet_base_v2_embedding(sentence):
    return model_all_mpnet_base_v2.encode(sentence)

def get_distilbert_base_v2_embedding(sentence):
    return model_sent_distbert_nli.encode(sentence)

def get_alberta_embedding(sentence):
    return model_alberta.encode(sentence)

def get_allenai_specter_embedding(sentence):
    inputs = specter_tokenize(sentence, padding=True, truncation=True, return_tensors="pt", max_length=5000)
    return specter_model(**inputs).last_hidden_state[:, 0, :]

@st.cache_resource
def load_data():
    data_all = dict()
    for model in model_list:
        data_all[model[0]] = dict()
        data_all[model[0]]['all'] = json.load(open(f'./Results/{model[0]}/all.json'))
        data_all[model[0]]['background'] = json.load(open(f'./Results/{model[0]}/background.json'))
        data_all[model[0]]['method'] = json.load(open(f'./Results/{model[0]}/method.json'))
        data_all[model[0]]['result'] = json.load(open(f'./Results/{model[0]}/result.json'))
    
    return data_all

data_all = load_data()


# for normal execution
@st.cache_data
def custom_docs_normal(facet, ATK, user_query):
    print("computing")
    print(type(model_sent_bert_nli))
    ens = dict()
    for model in model_list:
        try:
            method = model[0]
            weight = model[1]
            query_embedding = []
            if method == 'bert_nli':
                query_embedding = np.array(get_bert_nli_embedding(user_query)).tolist()
            elif method == 'bert_pp':
                query_embedding = np.array(get_bert_pp_embedding(user_query)).tolist()
            elif method == 'all_mpnet_base_v2':
                query_embedding = np.array(get_all_mpnet_base_v2_embedding(user_query)).tolist()
            elif method == 'distilbert_nli':
                query_embedding = np.array(get_distilbert_base_v2_embedding(user_query)).tolist()
            elif method == 'alberta':
                query_embedding = np.array(get_alberta_embedding(user_query)).tolist()
            else:
                query_embedding = get_allenai_specter_embedding(" ".join(user_query)).detach().numpy().tolist()[0]
            data = data_all[model[0]][facet]
            for id in data:
                if id not in ens:
                    ens[id] = 0
                ens[id] += cosine_sim(query_embedding, data[id]) * weight
        except:
            pass
        print("model done")
    sorted_results = sorted(ens.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    print("done")
    print(sorted_results[:ATK])
    return sorted_results[:ATK]

class get_doc:
    def __init__(self, paper_id, metadata, title, abstract, pred_labels_truncated, pred_labels):
      self.paper_id = paper_id
      self.metadata = metadata
      self.title = title
      self.abstract = abstract
      self.pred_labels_truncated = pred_labels_truncated
      self.pred_labels = pred_labels

docs = {}

with jsonlines.open('./data/abstracts-csfcube-preds.jsonl') as doc:
  for section in doc:
    docs[section['paper_id']] = get_doc(section['paper_id'], section['metadata'], section['title'], section['abstract'], section['pred_labels_truncated'], section['pred_labels'])

mapping = {
    'background':['background_label', 'objective_label'],
     'method':['method_label'],
     'result':['result_label']
}

def get_document(result):
    docu = {}
    docu['did'] = result[0]
    docu['title'] = docs[result[0]].title,  
    docu['score'] = result[1]
    docu['authors'] = []

    for author in docs[result[0]].metadata['authors']:
        name = []
        name.append(author['first'])
        name = name + author['middle']
        name.append(author['last'])
        auth = " ".join(name)
        docu['authors'].append(auth)
      
    docu['authors'] = " | ".join(docu['authors'])


    docu['year'] = docs[result[0]].metadata['year']
    docu['doi'] = docs[result[0]].metadata['doi']
    docu['venue'] = docs[result[0]].metadata['venue']

    abstract = []

    if custom_facet == 'all':
        for j in range(len(docs[result[0]].abstract)):
            abstract.append(docs[result[0]].abstract[j])
        # docu['abstract'] = " ".join(docs[result[0]].abstract)

    else:
        for j in range(len(docs[result[0]].abstract)):
            for l in mapping[custom_facet]:
              if docs[result[0]].pred_labels[j] == l:
                abstract.append(docs[result[0]].abstract[j])
    
    docu['abstract'] = " ".join(abstract)
    return docu

def make_result(docu):
    exp = st.expander(docu['title'][0] + " [ Ensembled score: " + str(docu['score']) + " ]")
    with exp:
        st.write("Title: ", docu['title'][0])
        st.write("Authors: ", docu['authors'])
        st.write("Abstract: ", docu['abstract'])
        if docu['doi'] is not None:
            st.write("URL (doi):", "https://doi.org/" + (docu['doi']))
        if docu['year'] is not None:
            st.write("Year: ", docu['year'])
        if docu['venue'] is not None:
            st.write("Venue: ", docu['venue'])




# st.title('SCIATICA')
st.markdown("<h1 style='text-align: center; color: cyan;'>SCIATICA</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: white;'>Research For Research Papers</h3>", unsafe_allow_html=True)
qdf = pd.read_csv('./queries-release.csv', sep=',')
facets = ['all', 'background', 'result', 'method']

sb = st.sidebar
with sb:
    st.subheader("Group 4 \n(CS60092: Information Retrieval, 2023 Spring)")
    st.write("Ashwani Kumar Kamal - 20CS10011")
    st.write("Hardik Pravin Soni - 20CS30023")
    st.write("Shiladitya De - 20CS30061")
    st.write("Sourabh Soumyakanta Das - 20CS30051")


custom_facet = st.selectbox("Select Facet", facets)
custom_query = st.text_input("Enter Query")
col1, col2 = st.columns(2)
with col1:
    bsearch = st.button("search")
with col2:
    no_results = st.slider("number of results", 1, 10, 5)
doc_result = st.empty()


if bsearch:
    message = st.empty()
    
    # message.write("Searching For Results...")

    start = datetime.now()
    result = custom_docs_normal(custom_facet, no_results, custom_query)
    end = datetime.now()
    diff = (end - start).total_seconds() 
    st.write("Fetched Results in", diff, "seconds")
    for res in result:
        docu = get_document(res)
        make_result(docu)
        # make_result(res[0], res[1], docs[res[0]].title, docs[res[0]].author, docs[res[0]].abstract, docs[res[0]].url)