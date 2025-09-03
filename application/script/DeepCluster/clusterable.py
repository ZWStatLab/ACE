import os, sys
import argparse


subf = open('submit.py','w')
subf.write('import os\n')

cmd = 'Rscript clusterable.R dcl'

for ii in range(4, 100, 5):
    file = os.path.join('npfiles_val', 'pro_output_{}.npz'.format(ii))
    cmd += ' {}'.format(file)

job = 'dip_dcl'
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
outf.write('module load R/4.1.2\n')
outf.write(cmd)
outf.close()
subf.write('os.system("sbatch %s")\n' % jobName)
subf.close()


