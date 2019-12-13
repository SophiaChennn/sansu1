import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import tensorflow as tf
import numpy as np
from data import config as cfg
from model import video_model
from data.get_input import ucf_data
import time

class Detect(object):
    def __init__(self,model,data):
        self.model = model
        self.data = data
        self.images = tf.placeholder(tf.float32,[1,cfg.clip_length,cfg.image_size,cfg.image_size,3])
        self.logits1 = tf.placeholder(tf.float32,[1,2])
        self.labels = tf.placeholder(tf.float32,[1,2])
        self.logits = self.model.h2d(self.images)
        self.logits = tf.nn.softmax(self.logits)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits,1),tf.argmax(self.labels,1)),tf.float32))

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        #print(bn_moving_vars)
        var_list += bn_moving_vars
        self.saver = tf.train.Saver(var_list)
        self.weight_file = 'checkpoint1/model.ckpt-25000'

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if self.weight_file is not None:

            self.saver.restore(self.sess,self.weight_file)
            print('Restoring weight file from:%s'%self.weight_file)


    def compute_accuracy(self):

        accuracy = 0
        start = time.time()
        a = 0
        b = 0
        f = open('result.txt','w+')       
        for i in range(5377):
            #print(i)
            
            images,labels,imname = self.data.get()
            #print(images.shape)
            #f = open('result.txt','w+')    
            feed_dict = {self.images:images}
            logit = self.sess.run(self.logits,feed_dict = feed_dict)
            #print(labels.shape)
            labels = np.reshape(labels,(2))
            logit = np.reshape(logit,(2))
            index1 = np.argmax(labels)
            index2 = np.argmax(logit)
            score = np.max(logit)
            str1 = imname+'\t'+str(index1)+'\t'+str(index2)+'\t'+str(logit[0])+'\t'+str(logit[1])+'\n'
            f.write(str1)
            print(i)
           
            
            
         
            

        #print(accuracy / 1000.0)
        #end = time.time() - start
        #print(end)

def main():
    model1 =video_model(False)
    data = ucf_data('test')
    detector = Detect(model1,data)

    detector.compute_accuracy()

if __name__ == '__main__':
    main()
            
