import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import numpy as np

template_data_path = "/network/lustre/iss02/aramis/datasets/adni/caps/caps_v2021/subjects/sub-ADNI006S0498/ses-M00/deeplearning_prepare_data/image_based/pet_linear/sub-ADNI006S0498_ses-M00_task-rest_acq-fdg_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-cerebellumPons2_pet.pt"
template_data = torch.load(template_data_path)
template_data = template_data[:,4:164:2,8:200:2,8:168:2]

caps_info = pd.read_csv('/network/lustre/iss02/aramis/datasets/adni/caps/caps_v2021.tsv', sep='\t', 
                        error_bad_lines=False)[['participant_id', 'session_id', 'age', 'diagnosis', 'MMSE']].set_index(['participant_id','session_id'])

path_imaging_data = "/network/lustre/iss02/aramis/datasets/adni/caps/caps_v2021/subjects/"

sub_lst = sorted(os.listdir(path_imaging_data))
data_dict = {'data':torch.tensor(template_data), 'timepoints':torch.ones(1), 'labels':torch.IntTensor([1])}

number_of_patients = 3000
i = 0 

for sub in tqdm(sub_lst):
    
    if i == number_of_patients:
        break
        
    sub_id = sub[12:]
    sub_path = path_imaging_data+sub
    ses_lst = sorted(os.listdir(sub_path))
    
    i += 1
    for ses in ses_lst:
        ses_path = os.path.join(sub_path, ses)
        if 'pet_linear' in os.listdir(ses_path):
            filename = 'deeplearning_prepare_data/image_based/pet_linear/' + sub + '_' + ses + '_task-rest_acq-fdg_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-cerebellumPons2_pet.pt'
            ses_data_path = os.path.join(ses_path, filename)

            #if (caps_info.loc[(sub,ses)]['diagnosis'] in ['CN']):
            #if 'AD' in caps_info.loc[sub].values:

            # Load the data
            ses_torch = torch.load(ses_data_path)
            if ses_torch.isnan().any() or torch.tensor([caps_info.loc[(sub,ses)]['age']]).isnan():
                continue
            # Subsample
            ses_torch = ses_torch[:,4:164:2,8:200:2,8:168:2]
            # Normalize
            ses_torch = ses_torch/ses_torch.max()

            data_dict['data'] = torch.cat((data_dict['data'], ses_torch.float()))
            data_dict['timepoints'] = torch.cat((data_dict['timepoints'], torch.tensor([caps_info.loc[(sub,ses)]['age']])))
            data_dict['labels'] = torch.cat((data_dict['labels'], torch.tensor([int(sub_id)])))

# Threshold to have a lot of 0 values in the input
mask_threshold = data_dict['data']<(torch.tensor(3e-2))
data_dict['data'][mask_threshold] = 0
        
# Then add the (useless timepointss) and delete the template data
data_dict['data'], data_dict['timepoints'], data_dict['labels'] = data_dict['data'][1:], data_dict['timepoints'][1:], data_dict['labels'][1:]

already_seen = []
group = []
idx = -1

for i in range(len(data_dict['labels'])):
    if data_dict['labels'][i] not in already_seen:
        already_seen.append(data_dict['labels'][i])
        idx += 1
    group.append(idx)
    
data_dict['RID'] = data_dict['labels']
data_dict['labels'] = torch.tensor(group)

torch.save(data_dict, '/network/lustre/iss02/aramis/users/benoit.sautydechalon/miccai_2022/ADNI_PET')