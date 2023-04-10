from errno import ENOBUFS
import numpy as np
import multiprocessing

data_all = dict()
model_list = []

def cosine_similarity_core(doc1, doc2):
  a = np.asarray(doc1)
  b = np.asarray(doc2)
  cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

  return cos_sim

def custom_doc_core(inp):
    facet = inp[0]
    query_embedding = inp[1]
    ens = inp[2]
    model = inp[3]
    data = data_all[model[0]][facet]
    weight = model[1]
    temp = dict()
    for id in ens:
        temp[id] = cosine_similarity_core(query_embedding, data[id]) * weight
    return list(temp.items())

def main(data, model, facet, query_embedding, ens):
    data_all = data
    model_list = model
    with multiprocessing.Pool(2) as pool:
        temp = zip(*pool.map(custom_doc_core, [(facet, query_embedding[model_list[i][0]], ens, model_list[i]) for i in range(len(model_list))]))
        for i in temp:
            for t in i:
                ens[t[0]] += t[1]
    return ens
