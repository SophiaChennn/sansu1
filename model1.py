import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np
from data import config as cfg
from tensorflow.contrib import layers
from nets import inception_v2
from nets import inception_utils
slim = tf.contrib.slim


class video_model(object):

    def __init__(self,training):
        
        self.training = training
        self.batch_size = cfg.batch_size
        self.clip_len = cfg.clip_length
        self.image_size = 112
        self.kernel_regu = layers.l2_regularizer(0.001)
        self.flag = 0
        #self.images = tf.placeholder(tf.float32,[self.batch_size,self.clip_len,self.image_size,self.image_size,3])
        #self.h2d(self.images)
        
            
    def build_network(self,inputs):

        #conv1
        net = tf.layers.conv3d(inputs,45,[1,7,7],[1,2,2],padding='SAME',
                               kernel_regularizer=self.kernel_regu,
                               name = 'conv1_s')
        net = tf.layers.batch_normalization(net,training = self.training,name = 'conv1_bn1')
        net = tf.nn.relu(net)

        net = tf.layers.conv3d(net,64,[3,1,1],[1,1,1],padding='SAME',
                               kernel_regularizer=self.kernel_regu,
                               name = 'conv1_t')
        net = tf.layers.batch_normalization(net,training = self.training,name = 'conv1_bn2')
        net = tf.nn.relu(net)
        
        #conv2
        for i in range(2):
            net = self.add_r3d_block(net, 64, 64,name='conv2_'+str(i+1))
        
        #conv3
        net = self.add_r3d_block(net, 64, 128, down_sampling=True, name = 'conv3_1')
        for i in range(1):
            net = self.add_r3d_block(net, 128, 128,name = 'conv3_'+str(i+2))
        
        #conv4
        net = self.add_r3d_block(net, 128, 256, down_sampling=True, name = 'conv4_1')
        for i in range(1):
            net = self.add_r3d_block(net, 256, 256,name = 'conv4_'+str(i+2))
        

        #conv5
        net = self.add_r3d_block(net, 256, 512, down_sampling=True, name = 'conv5_1')
        for i in range(1):
            net = self.add_r3d_block(net, 512, 512,name = 'conv5_'+str(i+2))
       
       #average_pool
        shape = net.shape.as_list()
        print(shape)
        net = tf.layers.average_pooling3d(net,[shape[1],shape[2],shape[3]],[1,1,1],name = 'average_pool')
        

        #fc layer

        net = tf.reshape(net,[-1,512])

        net = tf.layers.dense(net,1024,kernel_regularizer=self.kernel_regu,name='fc7')


        net = tf.layers.dense(net,600,kernel_regularizer=self.kernel_regu,name='fc8')
 
        print(net)
        return net

    def h2d(self,inputs):
        arg_scope = inception_utils.inception_arg_scope()
        inputs = tf.transpose(inputs,[1,0,2,3,4])
        shape = inputs.shape
        with slim.arg_scope(arg_scope):
           
            reuse = False
            
            for i in range(shape[0]):
                net,endpoint = inception_v2.inception_v2(inputs[i], num_classes = None,is_training = self.training, global_pool = True,reuse = reuse)
                print('net:',net)
                reuse = True

                if i == 0:
                    output1 = tf.expand_dims(endpoint['Mixed_3c'],0)
                    print('expand_dim,output1:',output1.shape)
                    output2 = tf.expand_dims(net,0)
                    print('expand_dim,output2:',output2.shape)

                else:
                    output1 = tf.concat([output1,tf.expand_dims(endpoint['Mixed_3c'],0)],axis = 0)
                    print(output1.shape)
                    output2 = tf.concat([output2,tf.expand_dims(net,0)],axis = 0)
                    print(output2.shape)

        output1 = tf.transpose(output1,[1,0,2,3,4])
        print(output1.shape)
        output2 = tf.transpose(output2,[1,0,2,3,4])
        print(output2.shape)

        #3D net
        output1 = tf.layers.conv3d(output1,96,[1,1,1],[1,1,1],padding='SAME',
                               kernel_regularizer=self.kernel_regu,
                               name = 'change')
        print(output1.shape)
        output1 = tf.layers.batch_normalization(output1,training = self.training,name = 'change_bn')
        output1 = tf.nn.relu(output1)
        output1 = self.add_r3d_block(output1, 96, 128, down_sampling=False, name = 'conv3_x')
        print(output1.shape)
        output1 = self.add_r3d_block(output1, 128,256, down_sampling=True, name = 'conv4_x')
        print(output1.shape)
        output1 = self.add_r3d_block(output1, 256,512, down_sampling=True, name = 'conv5_x')
        print(output1.shape)
        output1 = tf.reduce_mean(output1,[1,2,3])
        print(output1.shape)

        output2 = tf.reduce_mean(output2,[1,2,3])
        print(output2.shape)
        output = tf.concat([output1,output2],axis = 1)
        print(output.shape)
        output = tf.layers.dense(output,2,kernel_regularizer=self.kernel_regu,name='fc_layer')

        #print(output)       
        return output        
               
        
    def add_spatial_temporal_conv(self, inputs, in_filters, out_filters, stride,name):

        n = 3 * in_filters * out_filters * 3 * 3
        n /= in_filters * 3 * 3 + 3 * out_filters
        n = int(n)

        out = tf.layers.conv3d(inputs,n,[1,3,3],[1,stride[1],stride[2]],padding='SAME',
                               kernel_regularizer=self.kernel_regu,
                               name = name + '_s')
        out = tf.layers.batch_normalization(out,training = self.training,name = name + '_bn')

        out = tf.nn.relu(out)

        out = tf.layers.conv3d(out,out_filters,[3,1,1],[stride[0],1,1],padding='SAME',
                               kernel_regularizer=self.kernel_regu,
                               name = name + '_t')
        return out

    def add_r3d_block(self,inputs,input_filters,num_filters,name,down_sampling):

        shortcut = inputs

        if down_sampling:
            use_striding = [2,2,2]

        else:
            use_striding = [1, 1, 1]

        out = self.add_spatial_temporal_conv(inputs,input_filters,num_filters,stride=use_striding,name = name + '_1')
        out = tf.layers.batch_normalization(out,training = self.training,name = name + '_bn1')
        out = tf.nn.relu(out)

        out = self.add_spatial_temporal_conv(out,num_filters,num_filters,stride=[1,1,1],name = name + '_2')
        out = tf.layers.batch_normalization(out,training = self.training,name = name + '_bn2')

        if (num_filters != input_filters) or down_sampling:
            shortcut = tf.layers.conv3d(shortcut, num_filters, [1, 1, 1], use_striding,name = name + '_sample')
            shortcut = tf.layers.batch_normalization(shortcut,training = self.training,name = name + '_bn3')

        out = out + shortcut

        out = tf.nn.relu(out)

        return out

    def compute_loss(self,logits,labels):

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        #tf.add_to_collection('losses', cross_entropy_mean)

        return cross_entropy_mean


def main():
    model = video_model(True)
    input = np.random.rand(10,16,224,224,3)
    output = model.h2d(input.astype(np.float32))
    print(output.shape)
if __name__ == '__main__':
    main()


        
