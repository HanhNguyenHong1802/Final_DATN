# import from data_csv import Test, Train, Validation
import pandas as pd
import os, shutil
def pushing_train_data(dir, type):
  with open(dir,'r') as csvfile:
    df = pd.read_csv(csvfile)
    for row in df:
      try:
        dir1 = 'training_samples/'+type+'/'+ row.label
        os.mkdir(dir1)
        dir2 = 'D:/DATN/archive/'+type+'/'+ row.id
        shutil.copytree(dir2, dir1)
      except:                                                                  
        print('duplicate')
pushing_train_data('trainin')
          

