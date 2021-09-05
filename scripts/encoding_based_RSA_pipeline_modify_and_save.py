#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 14:57:40 2021

@author: nmei
"""

import os
import numpy as np
import pandas as pd
from shutil import rmtree,copyfile
from itertools import product

template = 'encoding_based_RSA_pipeline.py'
working_dir = '../data/Searchlight'
subjects = os.listdir(working_dir)
model_names = ['vgg19','mobilenet','resnet50','fasttext','glove','word2vec']
conditions = ['read','reenact']
df_iteration = pd.DataFrame(list(product(subjects,model_names,conditions)),
                            columns = ['sub','model_name','condition'])

node = 1
core = 16
mem = 6 * node * core
cput = 24 * node * core

#############
scripts_folder = 'ENRSA'
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
        line = f'# bash folder\nscripts/{scripts_folder}\n'
        f.write(line)
        f.close()

copyfile('utils.py',f'{scripts_folder}/utils.py')

collections = []
for ii,row in df_iteration.iterrows():
    sub,model_name,condition = row
    new_scripts_name = os.path.join(scripts_folder,template.replace('.py',f'_{sub}_{model_name}_{condition}.py'))
    with open(new_scripts_name,'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "../" in line:
                    line = line.replace("../","../../")
                elif 'change sub' in line:
                    line = f'    idx_sub             = "{sub}"\n'
                elif 'change condition' in line:
                    line = f'    condition           = "{condition}"\n'
                elif 'change model_name' in line:
                    line = f'    model_name          = "{model_name}"\n'
                elif "change n_jobs" in line:
                    line = '    n_jobs              = -1\n'
                elif "change alpha" in line:
                    line = '    alpha_max           = 20\n'
                elif "change dir idx" in line:
                    line = line.replace('idx = 3','idx = 4')
                new_file.write(line)
            old_file.close()
        new_file.close()
    new_batch_script_name = os.path.join(scripts_folder,f'enRSA{ii}')
    content = f"""#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={node}:ppn={core}
#PBS -l mem={mem}gb
#PBS -l cput={cput}:00:00
#PBS -N enRSA{ii}
#PBS -o outputs/out_{ii}.txt
#PBS -e outputs/err_{ii}.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd
echo {new_scripts_name.split('/')[-1]}
python "{new_scripts_name.split('/')[-1]}"
"""
    print(content)
    with open(new_batch_script_name,'w') as f:
        f.write(content)
        f.close()
    collections.append(f"qsub enRSA{ii}")

with open(f'{scripts_folder}/qsub_jobs.py','w') as f:
    f.write("""import os\nimport time""")

with open(f'{scripts_folder}/qsub_jobs.py','a') as f:
    for ii,line in enumerate(collections):
        if ii == 0:
            f.write(f'\nos.system("{line}")\n')
        else:
            f.write(f'time.sleep(.3)\nos.system("{line}")\n')
    f.close()