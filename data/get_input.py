import os
import numpy as np
import cv2
import pickle
from data import config as cfg
import time

class ucf_data(object):
    def __init__(self,phase):

        if phase == 'test':
            self.batch_size = 1
        else:
            self.batch_size = cfg.batch_size * cfg.num_gpus
        #self.batch_size = 1
        self.phase = phase
        self.clip_len = cfg.clip_length
        self.resize_height = 256
        self.resize_width = 256
        self.crop_size = cfg.image_size
        self.cursor = 0
        self.gt_labels = self.prepare()
        

    def prepare(self):

        if self.phase == 'train':
            with open('/data/2/chenyao/sansu/sansu-hongjun/data/train1.pkl','rb') as f:
                gt_labels = pickle.load(f)
                np.random.shuffle(gt_labels)
                print(len(gt_labels))

    
        else:

            with open('/data/2/chenyao/sansu/sansu-hongjun/data/test_video_cmp.pkl','rb') as f:
                gt_labels = pickle.load(f)
                np.random.shuffle(gt_labels)
                print(len(gt_labels))


        return gt_labels

    def get(self):
        
        images = np.zeros((self.batch_size,self.clip_len,self.crop_size,self.crop_size,3),np.float32)
        labels = np.zeros((self.batch_size,2),np.float32)
        imname = []
        count = 0
        while count < self.batch_size:
            image_path = self.gt_labels[self.cursor]['path']
            #image_path = image_path1.replace('/home/hongjun','/data/2/Hongjun/Sansu')
           
            imname = self.gt_labels[self.cursor]['imname']
            #imnames.append(imname)
            #print(image_path)
            #print(imname)
            if len(os.listdir(image_path)) < 20:
                self.cursor = self.cursor + 1

                if self.cursor >= len(self.gt_labels):
                    np.random.shuffle(self.gt_labels)
                    self.cursor = 0
                continue

            buffle = self.crop(image_path,imname,self.clip_len,self.crop_size)
            images[count] = buffle
            labels[count] = self.gt_labels[self.cursor]['label']
            count = count + 1
            self.cursor = self.cursor + 1
            
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0

        return images,labels,imname


    def crop(self,image_path,imname,clip_len,crop_size):

        #print(buffle.shape)
        buffle = np.zeros((clip_len,self.resize_height,self.resize_width,3),np.float32)
        image_num = len(os.listdir(image_path))
        time_index = image_num / clip_len
        for i in range(clip_len):
            
            if self.phase == 'train':
                flag = np.random.randint(int(i*time_index),int((i+1)*time_index)) + 1
            else:
                flag = int((2*i+1)*time_index / 2.0) + 1

            image_name = '%s-%05d.jpg'%(imname,flag)
            frame_path = os.path.join(image_path,image_name)
            image = cv2.imread(frame_path)
            image = cv2.resize(image,(self.resize_height,self.resize_width))
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            buffle[i] = image

        
        if self.phase == 'train':
    
            height_index = np.random.randint(buffle.shape[1] - crop_size)
            width_index = np.random.randint(buffle.shape[2] - crop_size)

        else:
            height_index = int((buffle.shape[1] - crop_size) / 2.0)
            width_index = int((buffle.shape[2] - crop_size) / 2.0)

        buffle = buffle[:,height_index:height_index + crop_size,width_index:width_index + crop_size,:]
        buffle = self.normalize(buffle)

        return buffle

    def normalize(self,buffle):

        buffle = (buffle / 255.) * 2.0 - 1.0

        return buffle

def main():

    data = ucf_data('test')
    #start = time.time()
    for i in range(3868):
        start = time.time()
        images,labels = data.get()
        t = time.time() - start
        print('%d    %.5f'%(i,t))
        #print(imname)
        #print(np.argmax(labels,1))
        #print(labels.shape)
        #end = time.time() - start
        #start = time.time()
        #print(end)

if __name__ == '__main__':
    main()
            

        
