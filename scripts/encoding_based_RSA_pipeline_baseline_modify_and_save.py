#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 02:44:20 2021

@author: nmei
"""

import os
import itertools
import numpy as np
import pandas as pd
from shutil import rmtree

template = 'encoding_based_RSA_pipeline_baseline.py'
working_dir = '../data/Searchlight'
subjects = os.listdir(working_dir)
cv_model_names = ['vgg19','mobilenet','resnet50']
w2v_model_names = ['fasttext','glove','word2vec']
conditions = ['read','reenact']

node = 1
core = 16
mem = 4 * node * core
cput = 24 * node * core

temp = np.array(list(itertools.product(*[subjects,conditions,])))
temp = pd.DataFrame(temp,columns = ['subject','condition'])
df = dict(subject = [],
          condition = [],
          cv_model_name = [],
          w2v_model_name = [],)
for ii,row in temp.iterrows():
    for cv_model_name,w2v_model_name in zip(cv_model_names,w2v_model_names):
        # it is not that they are paired, but we only need to run each model once,
        # and it happens that they all have 3 candidates
        [df[row_index].append(row_value) for row_index,row_value in zip(row.index,row)]
        df['cv_model_name'].append(cv_model_name)
        df['w2v_model_name'].append(w2v_model_name)
df = pd.DataFrame(df)
#############
scripts_folder = 'RSA_baseline'
if not os.path.exists(scripts_folder):
    os.mkdir(scripts_folder)
else:
    rmtree(scripts_folder)
    os.mkdir(scripts_folder)
os.mkdir(f'{scripts_folder}/outputs')
# add to gitignore
with open ('../.gitignore','r') as f:
    temp = [f'scripts/{scripts_folder}' in line for line in f]
    f.close()
if np.sum(temp) == 0:
    with open('../.gitignore','a') as f:
        line = f'#bash folder\nscripts/{scripts_folder}\n'
        f.write(line)
        f.close()

add_on = """from shutil import copyfile
copyfile('../utils.py','utils.py')
"""
collections = []
for ii,row in df.iterrows():
    src = '_{}_{}_{}_{}'.format(*list(row.to_dict().values()))
    new_scripts_name = os.path.join(scripts_folder,template.replace('.py',f'{src}.py'))
    with open(new_scripts_name,'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "../" in line:
                    line = line.replace("../","../../")
                elif 'sub                 = ' in line:
                    line = line.replace('123',row['subject'])
                elif 'cv_model_name       = ' in line:
                    line = line.replace('vgg19',row['cv_model_name'])
                elif 'w2v_model_name      = ' in line:
                    line = line.replace('fasttext',row['w2v_model_name'])
                elif 'condition           = ' in line:
                    line = line.replace('read',row['condition'])
                elif "# addon" in line:
                    line = "{}\n".format(add_on)
                new_file.write(line)
            old_file.close()
        new_file.close()
    new_batch_script_name = os.path.join(scripts_folder,f'RSA{ii+1}')
#    content = """#!/bin/bash
##$ -q long.q
##$ -N S{row['subject']}_{row['condition']}_{row['cv_model_name'][0]}_{row['w2v_model_name'][0]}
##$ -o output/out_sub{row['subject']}_{row['condition']}_{row['cv_model_name'][0]}_{row['w2v_model_name'][0]}.txt
##$ -e output/err_sub{row['subject']}_{row['condition']}_{row['cv_model_name'][0]}_{row['w2v_model_name'][0]}.txt
##$ -cwd
##$ -m be
##$ -M nmei@bcbl.eu
##$ -S /bin/bash
#
#pwd
#echo "{row['subject']}_{row['condition']}_{row['cv_model_name']}_{row['w2v_model_name']}"
#module load python/python3.6
#
#python "{created_file_name}"
#"""
    content = f"""#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={node}:ppn={core}
#PBS -l mem={mem}gb
#PBS -l cput={cput}:00:00
#PBS -N RSA{ii+1}
#PBS -o outputs/out_{ii+1}.txt
#PBS -e outputs/err_{ii+1}.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd
echo {new_scripts_name.split('/')[-1]}
python "{new_scripts_name.split('/')[-1]}"
    """
    with open(new_batch_script_name,'w') as f:
        f.write(content)
        f.close()
    collections.append(f"qsub RSA{ii+1}")

with open(f'{scripts_folder}/qsub_jobs.py','w') as f:
    f.write("""import os\nimport time""")

with open(f'{scripts_folder}/qsub_jobs.py','a') as f:
    for ii,line in enumerate(collections):
        if ii == 0:
            f.write(f'\nos.system("{line}")\n')
        else:
            f.write(f'time.sleep(3)\nos.system("{line}")\n')
    f.close()
