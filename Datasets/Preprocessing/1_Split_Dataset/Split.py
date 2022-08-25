# %%
##############################################
# Construction FVRID BaseDataset
# IMG: FVRID name
# NAME: Image name
# SOURCE: Source dataset
# SAVE: Target path
# ORI_NAME: Source dataset image name
##############################################
import os, sys
import pandas as pd
import shutil

# read label file
FVRID_CSV = pd.read_csv(r'./FVRID_Label.csv')

# source dirs
def check_dir(dir):
    if not os.path.isdir(dir):
        print("Alert! ", dir, " data source is not complete!")
VERIWILD_dir = r'../../VeriWild/images_all/'
VEHICLE1M_dir = r'../../Vehicle1M/image/'

# create save folder
SAVE_FOLDER = ['../../FVRID_syn/train_clear', '../../FVRID_syn/query_clear', '../../FVRID_syn/gallery_clear',
               '../../FVRID_real/train_foggy', '../../FVRID_real/query_foggy', '../../FVRID_real/gallery_foggy']
for folder in SAVE_FOLDER:
    if not os.path.isdir(folder):
        os.makedirs(folder)

# split as Veriwild and Vehicle1M
# columns = ['IMG', 'NAME', 'SOURCE', 'SAVE', 'ORI_NAME']
for idx in FVRID_CSV.index:
    print('%d/%d'%(idx, len(FVRID_CSV)), end='\r')
    data = FVRID_CSV.loc[idx].values
    if data[2] == 'Veriwild':
        src_ = os.path.join(VERIWILD_dir, data[4])
        dst_ = os.path.join("../../"+data[3], data[0])  # please check your path
        shutil.copy(src_, dst_)
    else:
        src_ = os.path.join(VEHICLE1M_dir, data[4])
        dst_ = os.path.join("../../"+data[3], data[0])  # please check your path
        shutil.copy(src_, dst_)      

print("Finish!")  
