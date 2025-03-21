import h5py
import numpy as np
import os, sys
import time
import argparse




subf = open('submit.py','w')
subf.write('import os\n')

datasets = ['USPS', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']

for task in ['DEPICT_hyper', 'DEPICT_num']:
    for eval_data in datasets:
        #with open(os.path.join('file_list', 'DEPICT_hyper', "{}.txt".format(eval_data)), "r") as file:
        #with open(os.path.join('file_list', 'DEPICT_num', "{}.txt".format(eval_data)), "r") as file:
        with open(os.path.join('file_list', task, "{}.txt".format(eval_data)), "r") as file:
            modelFiles = [line.strip() for line in file.readlines()]
        print(modelFiles)
        #modelFiles = [m[5:-4] for m in modelFiles]
        cmd = 'Rscript clusterable_DEPICT.R {} {}'.format(task, eval_data)
        for m in modelFiles:
            cmd += ' {}'.format(m)
        # print(cmd)
        # os.system(cmd)
        # while not os.path.isfile('dip_{}.npz'.format(eval_data)):
        #     time.sleep(0.1)
        job = 'dip{}_{}'.format(task, eval_data)
        jobName = job + '.sh'
        outf = open(jobName,'w')
        outf.write('#!/bin/bash\n')
        outf.write('\n')
        outf.write('#SBATCH --partition=stats_medium\n') 
        outf.write('#SBATCH --nodes=1 --mem=32G --time=24:00:00\n')
        outf.write('#SBATCH --ntasks=1\n')
        outf.write('#SBATCH --cpus-per-task=2\n')
        outf.write('#SBATCH --output=slurm-%A.%a.out\n')
        outf.write('#SBATCH --error=slurm-%A.%a.err\n')
        outf.write('#SBATCH --mail-type=ALL\n')
        outf.write('\n')
        outf.write('module load R/4.2.1\n')
        outf.write(cmd)
        outf.close()
        subf.write('os.system("sbatch %s")\n' % jobName)
subf.close()


