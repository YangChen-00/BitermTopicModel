import numpy as np
import pickle
import pyLDAvis
from biterm.btm import oBTM 
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary # helper functions

import pandas as pd
def __num_dist_rows__(array, ndigits=2):
    return array.shape[0] - int((pd.DataFrame(array).sum(axis=1) < 0.999).sum())


if __name__ == "__main__":

    # with open('data/data_new.pkl', 'rb') as fd:
    #     data = pickle.load(fd)
    # doc = []
    # with open('data/doc_new.txt', 'rb') as fd:
    #     for line in fd.readlines():
    #         doc.append(line)

    texts = open('data/after_preprocess_dataset_clean_english_only_new.txt').read().splitlines() # path of data file

    # txt中没有['pt']
    # texts = [' '.join(d['pt']) for d in data]
    # texts = [' '.join(d) for d in doc]

    # # 可视化会报错，报错的原因是没有去除掉长度为0的text
    # # Problem: https://blog.csdn.net/Accelerato/article/details/113444114
    # texts = [t for t in texts if len(t)>1]

    # vectorize texts
    vec = CountVectorizer(stop_words='english')
    X = vec.fit_transform(texts).toarray()

    # get vocabulary
    vocab = np.array(vec.get_feature_names())

    # get biterms
    biterms = vec_to_biterms(X)

    # with open('btm_model_20230410_debug.pkl', 'rb') as fd:
    #     btm = pickle.load(fd)
    
    # with open('btm_topics_20230410_debug.pkl', 'rb') as fd:
    #     topics = pickle.load(fd)
    

    # # create btm
    btm = oBTM(num_topics=20, V=vocab)

    # biterms_chunk = biterms[0:100]
    # btm.fit(biterms_chunk, iterations=50)

    print("\n\n Train Online BTM ..")
    for i in range(0, len(biterms), 100): # prozess chunk of 200 texts
        print(f"bitems: {i}/{len(biterms)}")
        biterms_chunk = biterms[i:i + 100]
        btm.fit(biterms_chunk, iterations=5) 
    topics = btm.transform(biterms)
    print(topics.shape)
    print(__num_dist_rows__(topics))

    with open('models/btm_model_04221358_5iter_debug.pkl', 'wb') as fd:
        pickle.dump(btm, fd)
    with open('models/btm_topics_04221358_5iter_debug.pkl', 'wb') as fd:
        pickle.dump(topics, fd)

    print("\n\n Visualize Topics ..")
    
    # 可视化会报错，报错的原因是没有去除掉长度为0的text
    topics = topics / topics.sum(axis=1)[:, None]
    print(__num_dist_rows__(topics))
    vis = pyLDAvis.prepare(btm.phi_wz.T, topics, np.count_nonzero(X, axis=1), vocab, np.sum(X, axis=0), mds='mmds')
    pyLDAvis.save_html(vis, './vis/online_btm_04221358_5iter.html')  # path to output

    print("\n\n Topic coherence ..")
    topic_summuary(btm.phi_wz.T, X, vocab, 10, "output/topic_coherence_result_04221358_5iter.txt")

    save_str = "output/topic_result_04221358_5iter.txt"
    result_str = ""
    print("\n\n Texts & Topics ..")
    for i in range(len(texts)):
        result_str += "{} (topic: {})\n".format(texts[i], topics[i].argmax())
        print("{} (topic: {})".format(texts[i], topics[i].argmax()))
    
    wf = open(save_str, 'w')
    wf.write(result_str)