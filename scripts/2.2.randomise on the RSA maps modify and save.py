#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 02:44:20 2021

@author: nmei
"""

import os
import numpy as np
import pandas as pd
from shutil import rmtree,copyfile

template = '2.1.randomise on the RSA maps.py'
working_dir = '../data/Searchlight'
subjects = os.listdir(working_dir)
cv_model_names = ['vgg19','mobilenet','resnet50']
w2v_model_names = ['fasttext','glove','word2vec']
conditions = ['read','reenact']

node = 1
core = 12
mem = 4
cput = 24

#############
scripts_folder = 'randomise'
if not os.path.exists(scripts_folder):
    os.mkdir(scripts_folder)
else:
    rmtree(scripts_folder)
    os.mkdir(scripts_folder)
os.mkdir(f'{scripts_folder}/outputs')
copyfile('utils.py',f'{scripts_folder}/utils.py')
# add to gitignore
with open ('../.gitignore','r') as f:
    temp = [f'scripts/{scripts_folder}' in line for line in f]
    f.close()
if np.sum(temp) == 0:
    with open('../.gitignore','a') as f:
        line = f'#bash folder\nscripts/{scripts_folder}\n'
        f.write(line)
        f.close()
from shutil import copyfile
copyfile('utils.py',f'{scripts_folder}/utils.py')

collections = []
for ii in range(12):
    new_scripts_name = os.path.join(scripts_folder,template.replace('.py',f'_{ii}.py'))
    with open(new_scripts_name,'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "../" in line:
                    line = line.replace("../","../../")
                elif 'change idx' in line:
                    line = f'idx = {ii}\n'
                new_file.write(line)
            old_file.close()
        new_file.close()
    new_batch_script_name = os.path.join(scripts_folder,f'RSA{ii}')
    content = f"""#!/bin/bash
#$ -q long.q
#$ -N Ran{ii}
#$ -o outputs/out_{ii}.txt
#$ -e outputs/err_{ii}.txt
#$ -cwd
#$ -m be
#$ -M nmei@bcbl.eu
#$ -S /bin/bash

pwd
echo "randomise map {ii}"
module load python/python3.6 fsl/6.0.0

python "{new_scripts_name.split('/')[-1]}"
"""

#    content = f"""#!/bin/bash
##PBS -q bcbl
##PBS -l nodes={node}:ppn={core}
##PBS -l mem={mem*core*node}gb
##PBS -l cput={cput*core*node}:00:00
##PBS -N RSA{ii}
##PBS -o outputs/out_{ii}.txt
##PBS -e outputs/err_{ii}.txt
#
#module load FSL/6.0.0-foss-2018b
#export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
#source activate keras-2.1.6_tensorflow-2.0.0
#cd $PBS_O_WORKDIR
#
#pwd
#echo {new_scripts_name.split('/')[-1]}
#python "{new_scripts_name.split('/')[-1]}"
#    """
    
#    content = f"""#!/bin/bash
##SBATCH --partition=regular
##SBATCH --job-name=RSA{ii}
##SBATCH --cpus-per-task={core}
##SBATCH --nodes={node}
##SBATCH --ntasks-per-node=1
##SBATCH --time={cput}:00:00
##SBATCH --mem-per-cpu={mem}G
##SBATCH --output=outputs/out_{ii}.txt
##SBATCH --error=outputs/err_{ii}.txt
##SBATCH --mail-user=nmei@bcbl.eu
#
#source /scratch/ningmei/anaconda3/etc/profile.d/conda.sh
#source activate keras-2.1.6_tensorflow-2.0.0
#module load FSL/6.0.0-foss-2018b
#cd $SLURM_SUBMIT_DIR
#
#pwd
#echo {new_scripts_name.split('/')[-1]}
#python "{new_scripts_name.split('/')[-1]}"
#    """
    print(content)
    with open(new_batch_script_name,'w') as f:
        f.write(content)
        f.close()
    collections.append(f"sbatch RSA{ii}")

with open(f'{scripts_folder}/qsub_jobs.py','w') as f:
    f.write("""import os\nimport time""")

with open(f'{scripts_folder}/qsub_jobs.py','a') as f:
    for ii,line in enumerate(collections):
        if ii == 0:
            f.write(f'\nos.system("{line}")\n')
        else:
            f.write(f'time.sleep(.3)\nos.system("{line}")\n')
    f.close()
