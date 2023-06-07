import numpy as np
import pickle
import pyLDAvis
import time
import pandas as pd
import argparse

from biterm.btm import oBTM 
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary # helper functions
from utils.logger import get_logger
from itertools import combinations

# init logger
timestamp = "{}".format(str(time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())))
log_save_dir = "./log/{}_FindTopicsNum.log".format(timestamp)
logger = get_logger(log_save_dir)
    
def compute_perplexity(btm, valid_biterms):
    logger.info("Computing Perplexity...")

    # B_d的长度为documents数，其中每个documents中有个一维数组存放生成的biterms
    for i, d in enumerate(valid_biterms): # 遍历每个doc的biterms，其中d为biterms组成的数组
        P_zb = np.zeros([len(d), btm.K]) # shape(num_bitems in doc[i], num_topics)
        for j, b in enumerate(d): # 取出其中的一个biterm
            # theta_z - (K,)
            # phi_wz - (num_words, K)
            
            P_zbi = btm.theta_z * btm.phi_wz[b[0], :] * btm.phi_wz[b[1], :]
            P_zb[j] = P_zbi / P_zbi.sum()
            
    perplexity = np.exp(-(np.log(P_zb).sum(axis=1)).mean(axis=0))
    
    return perplexity

#! 有BUG，出现INF
def compute_Griffiths2004(btm, valid_biterms):
    logger.info("Computing Griffiths2004...")
    
    P_zd = np.zeros([len(valid_biterms), btm.K]) # shape(num_docs, num_topics)
    for i, d in enumerate(valid_biterms): # 遍历每个doc的biterms，其中d为biterms组成的数组
        
        P_zb = np.zeros([len(d), btm.K]) # shape(num_bitems in doc[i], num_topics)
        for j, b in enumerate(d): # 取出其中的一个biterm
            # P(z,b)
            P_zbi = btm.theta_z * btm.phi_wz[b[0], :] * btm.phi_wz[b[1], :]
            P_zb[j] = P_zbi / P_zbi.sum()

        # 当doc中只有一个词时无法组成biterm
        if len(d) != 0:
            P_zd[i] = P_zb.sum(axis=0) / P_zb.sum(axis=0).sum()
        else:
            P_zd[i] = np.full((btm.K, ), 1.0 / btm.K)

    loglikes = []
    for i, d in enumerate(valid_biterms):
        inner_sum = 0.0
        for j, b in enumerate(d): # 取出其中的一个biterm
            P_bz = btm.phi_wz[b[0], :] * btm.phi_wz[b[1], :]
            inner = np.sum(P_zd[i] * P_bz)
            inner_sum += np.log(inner)
        loglikes.append(inner_sum)
    
    ll_med = np.median(loglikes)
    score = float(ll_med - np.log(np.mean(np.exp([-x for x in loglikes] + ll_med))))
    
    return score

def compute_CaoJuan2009(btm, valid_biterms):
    logger.info("Computing CaoJuan2009...")
    
    # matrix M1 topic-word
    m1 = np.exp(btm.phi_wz).T
    
    pairs = list(combinations([i for i in range(btm.K)], 2))
    
    dist_list = []
    for pair in pairs:
        x = m1[pair[0], :]
        y = m1[pair[1], :]
        
        dist = np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))
        dist_list.append(dist)
    
    score = np.sum(dist_list) / (btm.K * (btm.K - 1) / 2)
        
    return score

# def compute_Arun2010(btm, valid_biterms):
#     len = [len(biterms) for biterms in valid_biterms]
    
#     # matrix M1 topic-word
#     m1 = np.exp(btm.phi_wz).T
#     # u - Left singular vector matrix, same as u in R
#     # s - Singular Values, same as d in R
#     # vt - Right Singular Vector Matrix, same as v in R
#     m1_u, m1_s, m1_vt = np.linalg.svd(m1)
#     cm1 = np.matrix(m1_s)
    
#     P_zd = np.zeros([len(valid_biterms), btm.K]) # shape(num_docs, num_topics)
#     for i, d in enumerate(valid_biterms): # 遍历每个doc的biterms，其中d为biterms组成的数组
        
#         P_zb = np.zeros([len(d), btm.K]) # shape(num_bitems in doc[i], num_topics)
#         for j, b in enumerate(d): # 取出其中的一个biterm
#             # P(z,b)
#             P_zbi = btm.theta_z * btm.phi_wz[b[0], :] * btm.phi_wz[b[1], :]
#             P_zb[j] = P_zbi / P_zbi.sum()

#         # 当doc中只有一个词时无法组成biterm
#         if len(d) != 0:
#             P_zd[i] = P_zb.sum(axis=0) / P_zb.sum(axis=0).sum()
#         else:
#             P_zd[i] = np.full((btm.K, ), 1.0 / btm.K)
    
#     # matrix M2 document-topic
#     m2 = P_zd.T
#     cm2 = len * m2
#     norm <- norm(as.matrix(len), type="m")
#     cm2  = np.array(cm2 / norm)
#     divergence = np.sum(cm1 * np.log(cm1 / cm2)) + np.sum(cm2 * np.log(cm2/cm1))

def compute_Deveaud2014(btm, valid_biterms):
    # matrix M1 topic-word
    m1 = np.exp(btm.phi_wz).T
    
    pairs = list(combinations([i for i in range(btm.K)], 2))
    jsd_list = []
    for pair in pairs:
        x = m1[pair[0], :]
        y = m1[pair[1], :]
        
        jsd = 0.5 * np.sum(x * np.log(x / y)) + 0.5 * np.sum(y * np.log(y / x))
        jsd_list.append(jsd)
        
    score = np.sum(jsd_list) / (btm.K * (btm.K - 1))
        
    return score
    
def evaluation_metric(btm, valid_biterms, metric):
    if metric == 'perplexity':
        score = compute_perplexity(btm, valid_biterms)
    elif metric == 'Griffiths2004':
        score = compute_Griffiths2004(btm, valid_biterms)
    elif metric == 'CaoJuan2009':
        score = compute_CaoJuan2009(btm, valid_biterms)
    elif metric == "Deveaud2014":
        score = compute_Deveaud2014(btm, valid_biterms)
    return score

def partition_data(data_path, ratio):
    logger.info("-" * 10 + "Partitioning Data" + "-" * 10)
    texts = open(data_path).read().splitlines() # path of data file
    
    # vectorize texts
    logger.info("Vectorize texts ...")
    vec = CountVectorizer(stop_words='english')
    X = vec.fit_transform(texts).toarray()

    # get vocabulary
    logger.info("Get vocabulary ...")
    corpus_vocab = np.array(vec.get_feature_names())

    # get biterms
    logger.info("Get biterms ...")
    biterms = vec_to_biterms(X)
    
    segment_index = int(ratio * len(biterms))
    train_biterms = biterms[: segment_index]
    valid_biterms = biterms[segment_index : ]
    
    return train_biterms, valid_biterms, corpus_vocab

def find_topics_num(train_biterms, valid_biterms, corpus_vocab, interval, train_iter, metrics):
    logger.info("-" * 10 + "Find Topics Num" + "-" * 10)
    
    start = interval['from']
    end = interval['to']
    jump = interval['by']
    
    while start <= end:
        logger.info("-" * 10 + f"Training TopicsNum={start} BTM" + "-" * 10)
        btm = oBTM(num_topics=start, V=corpus_vocab)
        
        for i in range(0, len(train_biterms), 100): # prozess chunk of 200 texts
            logger.info(f"bitems: {i}/{len(train_biterms)}")
            biterms_chunk = train_biterms[i:i + 100]
            btm.fit(biterms_chunk, iterations=train_iter) 
        
        for metric in metrics:
            score = evaluation_metric(btm, valid_biterms, metric)
            logger.info(f'{metric} score: {score}')
            
        start += jump

if __name__ == "__main__":
    data_path = './data/2023-05-17-15-30-06_after_preprocess_dataset_clean_english_only_new.txt'
    train_biterms, valid_biterms, corpus_vocab = partition_data(data_path, 0.8)
    
    find_topics_num(train_biterms, valid_biterms, corpus_vocab,
                    {'from': 13, 'to': 20, 'by': 1}, 
                    1, 
                    ['perplexity', 'CaoJuan2009', 'Deveaud2014'])