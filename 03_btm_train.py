import numpy as np
import pickle
import pyLDAvis
from biterm.btm import oBTM 
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary # helper functions
from utils.logger import get_logger
import time
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--iteration_num',
                        help='the number of training',
                        required=True,
                        type=int)
    
    parser.add_argument('--topics_num',
                        help='the number of topics',
                        required=True,
                        type=int)
    
    parser.add_argument('--resume_path',
                        help='resume training',
                        type=str)
    
    args = parser.parse_args()
    
    return args

def __num_dist_rows__(array, ndigits=2):
    return array.shape[0] - int((pd.DataFrame(array).sum(axis=1) < 0.999).sum())

if __name__ == "__main__":
    args = parse_args()
    
    # init logger
    timestamp = "{}".format(str(time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())))
    log_save_dir = "./log/{}_BTM_Train.log".format(timestamp)
    logger = get_logger(log_save_dir)
    
    logger.info("Args: {}".format(args))
    postfix = '_' + timestamp + '_' + str(args.iteration_num) + 'iter' + '_' + str(args.topics_num) + "t"
    
    data_path = './data/2023-05-17-15-30-06_after_preprocess_dataset_clean_english_only_new.txt'
    logger.info("Read texts from {}".format(data_path))
    texts = open(data_path).read().splitlines() # path of data file

    # vectorize texts
    logger.info("Vectorize texts ...")
    vec = CountVectorizer(stop_words='english')
    X = vec.fit_transform(texts).toarray()

    # get vocabulary
    logger.info("Get vocabulary ...")
    vocab = np.array(vec.get_feature_names())

    # get biterms
    logger.info("Get biterms ...")
    biterms = vec_to_biterms(X)
    
    # create btm
    if args.resume_path:
        logger.info("Resume models...")
        btm = oBTM(num_topics=args.topics_num, V=vocab)
        f = open(args.resume_path, 'rb')
        reader = f.read()
        btm = pickle.loads(reader)
    else:
        btm = oBTM(num_topics=args.topics_num, V=vocab)
    
    logger.info("Train Online BTM ..")
    for i in range(0, len(biterms), 100): # prozess chunk of 200 texts
        logger.info(f"bitems: {i}/{len(biterms)}")
        biterms_chunk = biterms[i:i + 100]
        btm.fit(biterms_chunk, iterations=args.iteration_num) 
    topics = btm.transform(biterms)
    # logger.info(topics.shape)
    # logger.info(__num_dist_rows__(topics))

    save_btm_model_path = './models/btm_model{}.pkl'.format(postfix)
    save_btm_topics_path = './models/btm_topics{}.pkl'.format(postfix)
    with open(save_btm_model_path, 'wb') as fd:
        pickle.dump(btm, fd)
    with open(save_btm_topics_path, 'wb') as fd:
        pickle.dump(topics, fd)

    logger.info("Visualize Topics ..")

    topics = topics / topics.sum(axis=1)[:, None]
    # logger.info(__num_dist_rows__(topics))
    
    save_html_path = './vis/online_btm{}.html'.format(postfix)
    logger.info("Saving Vis Html to {} ..".format(save_html_path))
    vis = pyLDAvis.prepare(btm.phi_wz.T, topics, np.count_nonzero(X, axis=1), vocab, np.sum(X, axis=0), mds='mmds')
    pyLDAvis.save_html(vis, save_html_path)  # path to output

    save_topic_coherence_result_path = "./output/topic_coherence_result{}.txt".format(postfix)
    logger.info("Generating Topic coherence to {} ..".format(save_topic_coherence_result_path))
    topic_summuary(btm.phi_wz.T, X, vocab, 10, save_topic_coherence_result_path)

    save_topic_result_path = "./output/topic_result{}.txt".format(postfix)
    result_str = ""
    logger.info("Texts & Topics to {} ..".format(save_topic_result_path))
    for i in range(len(texts)):
        result_str += "{} (topic: {})\n".format(texts[i], topics[i].argmax())
        print("{} (topic: {})".format(texts[i], topics[i].argmax()))
    
    wf = open(save_topic_result_path, 'w')
    wf.write(result_str)