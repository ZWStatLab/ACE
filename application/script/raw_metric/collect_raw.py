import numpy as np
import sklearn
from sklearn import metrics
import os, h5py
import pickle as pk
from collections import defaultdict
import argparse
from numpy import linalg as LA
import math


if __name__ == '__main__':

    datasets_jule = ['USPS', 'UMist', 'COIL-20', 'COIL-100', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']
    datasets_depict = ['USPS', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']
    datasets_all = {
        'JULE_hyper': datasets_jule ,
        'JULE_num': datasets_jule ,
        'DEPICT_num': datasets_depict,
        'DEPICT_hyper': datasets_depict,
    }

    metric_list = ['ccc','dunn','cind', 'sdbw', 'ccdbw']


    for task in datasets_all.keys():
        datasets = datasets_all[task]

        for eval_data in datasets:
            scored = defaultdict(dict)
        
            with open(os.path.join('file_list', task, "{}.txt".format(eval_data)), "r") as file:
                modelFiles = [line.strip() for line in file.readlines()]

            for key in modelFiles:

                file = "{}/raw_tmp/rr_{}.npz".format(task, key)
                data = np.load(file)
                for metric in metric_list:
                    value = data[metric]
                    # handle abnormal function output values
                    if value == True:
                        print('cv',value)
                        value = np.nan
                    if value == -2147483648:
                        print('cv',value)
                        value = np.nan
                    if math.isinf(value):
                        print('cv',value)
                        value = np.nan
                    scored[metric][key] = value


            for metric in ['dav', 'ch', 'euclidean', 'cosine']:
                with open('{}/raw_metric/merge_{}_{}_score.pkl'.format(task, eval_data, metric), 'rb') as file:
                    scored2 = pk.load(file)
                for key, value in scored2.items():
                    print(len(value))
                    scored[key] = value

            #print(scored)
            with open('{}/raw_metric/merge_all_{}_score.pkl'.format(task, eval_data), 'wb') as file:
                pk.dump(scored, file)

            
