from array import array
import os
import string
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
import pandas as pd
import random
from tqdm import tqdm
def load_data(dirname:string, x_train:array, img_rows, img_cols):
    ls_path = os.path.join(dirname)
    listing = os.listdir(ls_path)
    for ls in tqdm(listing):
      listing_stop = sorted(os.listdir(os.path.join(ls_path,ls))) 
      frames = []
      img_depth=0
      for imgs in listing_stop:
        if img_depth <16:
          img = os.path.join(os.path.join(ls_path,ls),imgs)
          frame = cv2.imread(img)
          frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frames.append(gray)
          img_depth=img_depth+1
        else: break
      
      input_img = np.array(frames)
      ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
      ipt=np.rollaxis(ipt,2,0)
      x_train.append(ipt)

class DataLoader():
  def prepare_data(self, img_rows, img_cols):
    self.img_rows = img_rows
    self.img_cols = img_cols
    X_tr = []
    # ls_path = os.path.join("training_samples/Thumb Down")
    load_data("training_samples/Thumb Down", X_tr, img_rows, img_cols)
    load_data("training_samples/Thumb Up", X_tr, img_rows, img_cols)
    load_data("training_samples/Drumming Fingers", X_tr, img_rows, img_cols)
    load_data("training_samples/Sliding Two Fingers Right", X_tr, img_rows, img_cols)
    load_data("training_samples/Sliding Two Fingers Left", X_tr, img_rows, img_cols)
    load_data("training_samples/No gesture", X_tr, img_rows, img_cols)
    return X_tr

