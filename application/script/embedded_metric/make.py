import sys
import os
import pickle as pk



subff = open('submit.py','w')
subff.write('import os\n')


datasets_jule = ['USPS', 'UMist', 'COIL-20', 'COIL-100', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']
datasets_depict = ['USPS', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']
datasets_all = {
    'JULE_hyper': datasets_jule ,
    'JULE_num': datasets_jule ,
    'DEPICT_hyper': datasets_depict,
    'DEPICT_num': datasets_depict,
}

# change this line to move from step 1 to step 2
step = 1 #step = 2

# step one
if step == 1:
    for task in datasets_all.keys():
        datasets = datasets_all[task]
        for metric in ['dav', 'ch', 'euclidean', 'cosine']:
            for dataset in datasets:
                job = 'computeinternal_{}_{}_{}'.format(task, dataset, metric)
                jobName=job + '.sh'
                outf = open(jobName,'w')
                outf.write('#!/bin/bash\n')
                outf.write('\n')
                if dataset == 'COIL-100':
                    outf.write('#SBATCH --partition=stats_long\n') 
                else:
                    outf.write('#SBATCH --partition=stats_medium\n') 
                if dataset == 'COIL-100':
                    outf.write('#SBATCH --nodes=1 --mem=96G --time=24:00:00\n')
                else:
                    outf.write('#SBATCH --nodes=1 --mem=32G --time=24:00:00\n')
                outf.write('#SBATCH --ntasks=1\n')
                outf.write('#SBATCH --cpus-per-task=2\n')
                outf.write('#SBATCH --output=slurm-%A.%a.out\n')
                outf.write('#SBATCH --error=slurm-%A.%a.err\n')
                outf.write('#SBATCH --mail-type=ALL\n')
                outf.write('\n')
                outf.write('conda info --envs\n')
                outf.write('eval $(conda shell.bash hook)\n')
                outf.write('source ~/miniconda/etc/profile.d/conda.sh\n')
                outf.write('conda activate dcl\n')
                outf.write('python3 embedded_data.py --dataset {} --metric {} --task {} \n'.format(dataset, metric, task))
                outf.close()
                subff.write('os.system("sbatch %s")\n' % jobName)
    subff.close()
elif step == 2:
    # step 2
    for task in datasets_all.keys():
        subf = open('submit{}.py'.format(task),'w')
        subf.write('import os\n')
        datasets = datasets_all[task]
        for dataset in datasets:
            with open(os.path.join('file_list', task, "{}.txt".format(dataset)), "r") as file:
                modelFiles = [line.strip() for line in file.readlines()]
            for m1 in modelFiles:
                for m2 in modelFiles:
                    ff = '{}/tmp/rr_{}_{}.npz'.format(task, m1, m2)
                    if not os.path.isfile(ff):
                        job = 'Rmerge_{}_{}_{}_{}'.format(task, dataset, m1, m2)
                        jobName=job + '.sh'
                        outf = open(jobName,'w')
                        outf.write('#!/bin/bash\n')
                        outf.write('\n')
                        if dataset in ['USPS', 'MNIST-test', 'COIL-100', 'YTF']:
                            outf.write('#SBATCH --partition=stats_medium\n')
                            outf.write('#SBATCH --nodes=1 --mem=8G --time=24:00:00\n')
                        else:
                            outf.write('#SBATCH --partition=stats_short\n')
                            outf.write('#SBATCH --nodes=1 --mem=8G --time=1:00:00\n')
                        outf.write('#SBATCH --ntasks=1\n')
                        outf.write('#SBATCH --cpus-per-task=1\n')
                        outf.write('#SBATCH --output=slurm-%A.%a.out\n')
                        outf.write('#SBATCH --error=slurm-%A.%a.err\n')
                        outf.write('#SBATCH --mail-type=ALL\n')
                        outf.write('\n')
                        outf.write('module load R/4.1.2\n')
                        outf.write('Rscript embedded.r {} {} {}\n'.format(task, m1,m2))
                        outf.close()
                        subf.write('os.system("sbatch %s")\n' % jobName)
        subf.close()
