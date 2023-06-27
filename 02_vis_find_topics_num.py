import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib

def read_scores_to_list(find_topics_num_log_path):
    reader = open(find_topics_num_log_path, "r")
    
    perplexity_list, CaoJuan2009_list, Deveaud2014_list = [], [], []
    for line in reader.readlines():
        if "perplexity score" in line:
            perplexity_list.append(float(line.split(' ')[-1]))
        elif "CaoJuan2009 score" in line:
            CaoJuan2009_list.append(float(line.split(' ')[-1]))
        elif "Deveaud2014 score" in line:
            Deveaud2014_list.append(float(line.split(' ')[-1]))
            
    return perplexity_list, CaoJuan2009_list, Deveaud2014_list

def vis(vis_path, score_name, score_list):
    begin=2
    end=25
    plt.subplots(figsize=(12, 10))
    fontsize=15

    score_list = score_list[:end-begin+1]
    plt.plot(np.arange(begin, end+1), score_list, linewidth=5,
             marker='o', markersize='7', markeredgewidth=10)

    plt.xlabel("Number of Topics")
    plt.xticks(np.arange(2, end+1, 1))
    
    # set vertical dashed line
    plt.vlines(np.arange(2, end+1, 1), 
               np.min(score_list), np.max(score_list), 
               linestyles='dashed', colors='gray')

    if score_name == 'perplexity':
        plt.title("{} Score (log)".format(score_name), fontsize=fontsize)
        plt.yscale('log') # log yscale
        plt.xlabel("Number of Topics", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
    else:
        plt.title("{} Score".format(score_name))
        
    matplotlib.rcParams.update({'font.size': fontsize})
    plt.savefig(vis_path, bbox_inches='tight')
    plt.show()

# Cao and perplexity need to be minimized
# Deveaud2014 need to be maximized
def vis_scores(vis_root_path, timestamp, scores_name, scores_list):
    if isinstance(scores_name, list):
        for (s_name, s_list) in zip(scores_name, scores_list):
            vis_path = vis_root_path + os.sep + "{}_vis_{}_score.png".format(timestamp, s_name)
            vis(vis_path, s_name, s_list)
    elif isinstance(scores_name, str):
        vis_path = vis_root_path + os.sep + "{}_vis_{}_score.png".format(timestamp, s_name)
        vis(vis_path, s_name, s_list)

if __name__ == "__main__":
    timestamp = "2023-05-30-20-23-20"
    find_topics_num_log_path = "log/{}_FindTopicsNum.log".format(timestamp)
    perplexity_list, CaoJuan2009_list, Deveaud2014_list = read_scores_to_list(find_topics_num_log_path)
    
    vis_root_path = "analysis/vis/num_topics_metrics"
    vis_scores(vis_root_path, timestamp, 
               ['perplexity', 'CaoJuan2009', 'Deveaud2014'],
               [perplexity_list, CaoJuan2009_list, Deveaud2014_list])