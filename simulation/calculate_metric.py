import numpy as np
import sklearn
from sklearn import metrics
import os, h5py
import pickle as pk
from collections import defaultdict
import argparse
from numpy import linalg as LA
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score

def get_files_with_substr_suffix(dirpath, substr, suffix):
    '''
    get files with given substring and suffix
    '''
    all_files = os.listdir(dirpath)
    files = [file for file in all_files if substr in file and file.endswith(suffix)]
    return files

def clustering_accuracy(gtlabels, labels):
    gtlabels = np.array(gtlabels, dtype='int64')
    labels = np.array(labels, dtype='int64')
    cnt_matrix = []
    categories = np.unique(gtlabels)
    nr = np.amax(labels) + 1
    for i in np.arange(len(categories)):
      cnt_matrix.append(np.bincount(labels[gtlabels == categories[i]], minlength=nr))
    cnt_matrix = np.asarray(cnt_matrix).T
    row_ind, col_ind = linear_sum_assignment(np.max(cnt_matrix) - cnt_matrix)
    return float(cnt_matrix[row_ind, col_ind].sum()) / len(gtlabels)

def calinski_harabasz_score(x,y):
    if len(np.unique(y)) > 1:
        return metrics.calinski_harabasz_score(x,y)
    else:
        return None

def davies_bouldin_score(x,y):
    if len(np.unique(y)) > 1:
        return - metrics.davies_bouldin_score(x,y)
    else:
        return None

def silhouette_score(x,y, metric):
    if len(np.unique(y)) > 1:
        return metrics.silhouette_score(x, y, metric=metric)
    else:
        return None


def clustering_score(x,y, metric):
    if metric == 'dav':
        return davies_bouldin_score(x,y)
    elif metric == 'ch':
        return calinski_harabasz_score(x,y)
    else:
        return silhouette_score(x, y, metric=metric)



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', default='dense')
    parser.add_argument('--metric', default='euclidean')

    args = parser.parse_args()
    metric = args.metric
    modelpath = args.modelpath


    tpath = os.path.join(modelpath, 'embedded_metric')
    if not os.path.isdir(tpath):
        os.mkdir(tpath)

    features={}
    labels={}
    truths={}
    scored = defaultdict(dict)
    modelFiles = get_files_with_substr_suffix(modelpath, 'output', 'npz')

    for m in modelFiles:
        files = np.load(os.path.join(modelpath,m))
        features[m] = np.array(files['y_features'])
        labels[m] = np.squeeze(np.array(files['y_pred']))
        truths[m] = np.squeeze(np.array(files['truth']))

    for m in modelFiles:
        x = features[m]
        print(x.shape)
        for key in labels.keys():
            y = labels[key]
            score_local = clustering_score(x, y, metric=metric)
            scored[m][key] = score_local
            labelset = np.array(y)

    with open(os.path.join(tpath,'merge_{}_score.pkl'.format(metric)), 'wb') as file:
        pk.dump(scored, file)
    print(scored)

    ## load the raw data
    tpath = os.path.join(modelpath, 'raw_metric')
    if not os.path.isdir(tpath):
        os.mkdir(tpath)


    raw_file = os.path.join(modelpath, 'sim.npz')
    raw_files = np.load(raw_file)
    data, truth_y = raw_files['X'], raw_files['y']

    ## get the raw scores
    scored = defaultdict(dict)

    for key in modelFiles:
        y = labels[key]
        truth = truths[key]
        scored[metric][key] = clustering_score(data, y, metric=metric)
        scored['nmi'][key] = normalized_mutual_info_score(truth, y)
        scored['acc'][key] = clustering_accuracy(truth, y)

    with open(os.path.join(tpath,'merge_{}_score.pkl'.format(metric)), 'wb') as file:
        pk.dump(scored, file)
    print(scored)
    nmv = {}
    acv = {}
    for key in modelFiles:
        y = labels[key]
        truth = truths[key]
        nmv[key] = normalized_mutual_info_score(truth, y)
        acv[key] = clustering_accuracy(truth, y)

    nmv = dict(sorted(nmv.items(), key=lambda item: item[0]))
    acv = dict(sorted(acv.items(), key=lambda item: item[0]))

    tpath = os.path.join(modelpath, 'true.pkl')
    with open(tpath, 'wb') as ff:
        pk.dump([nmv, acv], ff)


