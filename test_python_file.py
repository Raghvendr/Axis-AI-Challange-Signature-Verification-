"""
 Team datascience solution submission
 
"""
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import cv2
from os import walk
from preprocessing import preprocess_image 
from os.path import join
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
from keras.models import Sequential, Model
from keras.optimizers import  RMSprop
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import model_from_json


def compute_accuracy_roc(predictions, labels):
   # Compute ROC accuracy with a range of thresholds on distances.
   dmax = np.max(predictions)
   dmin = np.min(predictions)
   nsame = np.sum(labels == 1)
   ndiff = np.sum(labels == 0)
   
   step = 0.01
   max_acc = 0
   
   for d in np.arange(dmin, dmax+step, step):
       idx1 = predictions.ravel() <= d
       idx2 = predictions.ravel() > d
       
       tpr = float(np.sum(labels[idx1] == 1)) / nsame       
       tnr = float(np.sum(labels[idx2] == 0)) / ndiff
       acc = 0.5 * (tpr + tnr)       
#       print ('ROC', acc, tpr, tnr)
       
       if (acc > max_acc):
           print(d)
           max_acc = acc
           
   return max_acc


## folder path  for the registration and  test data
   """
   
   we have considered the username has been mentioned in [3:6] index in columns
   you will have to assign folder names
   
   You need to give your inputs here
   
   """
register_user_folder =""
test_folder =" "





def pickle_images_labels(database_folder):
#    train_folder = 'DataBase'
#    test_folder = 'test'
    images_genuine = []
    labels_genuine = []
    name_genuine =[]
    for (dirpath, dirnames, filenames) in walk(database_folder):
        for filename in filenames:
            print(filename)
            if '.jpg' in filename:
                image_path = join(dirpath, filename)
                img = cv2.imread(image_path)
                preprocessed_image =preprocess_image(img)
                if np.any(img == None):
                    continue
                myList =  preprocessed_image.flatten()     
                images_genuine.append(myList)
                labels_genuine.append(int(filename[3:6]))
                name_genuine.append(filename)
                                
    return images_genuine,labels_genuine,name_genuine

images_genuine, labels_genuine, name_genuine = pickle_images_labels(register_user_folder)




images_oot, labels_oot, name_oot = pickle_images_labels(database_folder = test_folder)



images_genuine = np.array(images_genuine)
images_oot = np.array(images_oot)


images_genuine = images_genuine.astype('float32')
images_oot = images_oot.astype('float32')


images_genuine /= 255
images_oot /= 255



user_reg=images_genuine.reshape(-1,80,150,1)
user_check=images_oot.reshape(-1,80,150,1)


# load json and create model
json_file = open('./final_model/signature_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./final_model/signature_siamese.h5")
print("Loaded model from disk")


import pandas as pd

df = pd.DataFrame()
index =0

for j,k in enumerate(labels_genuine):
    print(j,k)
    
    for l,m in enumerate(labels_oot):
        print(l,m)
        if (k ==m):
            tmp1= user_reg[j]
            tmp2 = user_check[l]
            tmp1 = tmp1.reshape((1,80,150,1))
            tmp2 = tmp2.reshape((1,80,150,1))
            
            diff = loaded_model.predict([tmp1,tmp2])
            print(diff)
            
            df.loc[index,"difference"] = diff[0][0]
            df.loc[index,"userid"] = name_genuine[j]
            df.loc[index,"user_check"] = name_oot[l]
            
            index +=1

df.to_csv("results_2.csv")

mean_df = df.groupby(['user_check'])['difference'].mean().reset_index()

mean_df.to_csv("final_pdf.csv")

for i in range(len(mean_df)):
    if mean_df["difference"][i] >=0.10:
        mean_df.loc[i,"Forged"] ="Yes"
    else:
        mean_df.loc[i,"Forged"] ="No"
        
mean_df= mean_df[["user_check","Forged"]]
        

mean_df.to_csv("Team_datascience_final_output.csv")


            
            
            
        
        
        






         