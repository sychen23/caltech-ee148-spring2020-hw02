import numpy as np
import os
import random
import json

np.random.seed(2020) # to ensure you always get the same train/test split

user_profile = os.environ['HOME']

data_path = '%s/data/EE148/RedLights2011_Medium' % user_profile
gts_path = '%s/data/EE148/hw02_annotations' % user_profile
split_path = '%s/data/EE148/hw02_splits' % user_profile
os.makedirs(split_path, exist_ok=True) # create directory if needed

split_test = True # set to True and run when annotations are available

train_frac = 0.85

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# split file names into train and test
file_names_train = []
file_names_test = []
'''
Your code below. 
'''
random.shuffle(file_names)
num_file_names = len(file_names)
num_train_data = int(num_file_names * train_frac)
file_names_train = file_names[:num_train_data]
file_names_test = file_names[num_train_data:]

assert (len(file_names_train) + len(file_names_test)) == len(file_names)
assert len(np.intersect1d(file_names_train,file_names_test)) == 0

np.save(os.path.join(split_path,'file_names_train.npy'),file_names_train)
np.save(os.path.join(split_path,'file_names_test.npy'),file_names_test)

if split_test:
    with open(os.path.join(gts_path, 'formatted_annotations_students.json'),'r') as f:
        gts = json.load(f)
    
    # Use file_names_train and file_names_test to apply the split to the
    # annotations
    gts_train = {}
    gts_test = {}
    '''
    Your code below. 
    '''
    gts_train = {f:gts[f] for f in gts if f in file_names_train}
    gts_test = {f:gts[f] for f in gts if f in file_names_test}
    
    with open(os.path.join(gts_path, 'annotations_train.json'),'w') as f:
        json.dump(gts_train,f)
    
    with open(os.path.join(gts_path, 'annotations_test.json'),'w') as f:
        json.dump(gts_test,f)
    
    
