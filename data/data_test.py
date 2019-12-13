import os
import numpy as np
import cv2
import pickle

datapath = '/data/2/chenyao/sansu/test'
flag = 0
cache_file2 = 'test_video_cmp.pkl'
gt_labels = []
test_labels = []
t = 0
i = 0
for file in os.listdir(datapath):

    dirpath = os.path.join(datapath,file)
    if file == 'normal':
        flag = 1
    else:
        flag = 0

    
    #for file1 in os.listdir(dirpath):
    if True:
        dirpath1 = dirpath#os.path.join(dirpath,file)
        #print(dirpath1)
        for file2 in os.listdir(dirpath1):


            i = i + 1
         
            actionpath = os.path.join(dirpath1,file2)

            labels = np.zeros((2),np.float32)
            labels[flag] = 1.0


            num = len(os.listdir(actionpath))

            if num > 10:
         
                #a = np.random.rand()
                #if a <= 0.8:
                
                #    gt_labels.append({'path':actionpath,
                #                      'label':labels,
                #                      'imname':file2})
            
                #else:
                print(actionpath)
                test_labels.append({'path':actionpath,
                                    'label':labels,
                                    'imname':file2})
            else:
                #print(actionpath)
                pass
    
         
       



#np.random.shuffle(gt_labels)
np.random.shuffle(test_labels)

#print(len(gt_labels))
#print(len(test_labels))

with open(cache_file2,'wb') as f2:
    #print('start')
    pickle.dump(test_labels,f2)
    #print('done')
