
# coding: utf-8

# In[1]:

import tensorflow as tf
import cv2
import numpy as np
slim=tf.contrib.slim
import time
import os

from tensorflow.python.ops import control_flow_ops


# In[2]:

#残差网络
def convBlock(inputs,numOut,is_training,scope=''):
    with slim.arg_scope([slim.conv2d],padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),

                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training, 'decay': 0.95}):
        net=slim.conv2d(inputs,numOut//2,kernel_size=[1,1],scope= scope+'cb1')
        net=slim.conv2d(net,numOut//2,kernel_size=[3,3],scope= scope+'cb2')
        net=slim.conv2d(net,numOut,kernel_size=[1,1],scope= scope+'cb3')

        return net
def skipLayer(inputs,numOut,scope=''):
    numIn=inputs.shape[-1]
    if numIn==numOut:
        return inputs
    else:
        return slim.conv2d(inputs,numOut,kernel_size=[1,1],activation_fn=None,scope=scope+'sk')

def residual(inputs,numOut,is_training,scope=''):
    convb=convBlock(inputs,numOut,is_training,scope)
    skip=skipLayer(inputs,numOut,scope)
    return tf.add_n([convb,skip],name=scope+'re')


# In[3]:

#hourglass网络
def hourglass(inputs, n, numOut,is_training, name = 'hourglass'):
    """ Hourglass Module
    Args:
        inputs	: Input Tensor
        n		: Number of downsampling step
        numOut	: Number of Output Features (channels)
        name	: Name of the block
    """
    with tf.name_scope(name):
        # Upper Branch
        up_1 = residual(inputs, numOut,is_training,scope = str(n)+'up_1')
        # Lower Branch
        low_ = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID')
        low_1= residual(low_, numOut,is_training,scope = str(n)+'low_1')

        if n > 0:
            low_2 =hourglass(low_1, n-1, numOut,is_training, name = str(n)+'low_2')
        else:
            low_2 =residual(low_1, numOut,is_training,scope =str(n)+ 'low_2')

        low_3 = residual(low_2, numOut, is_training,scope = str(n)+'low_3')
        up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3]*2, name =str(n)+ 'upsampling')
        
        return tf.add_n([up_2,up_1], name=name+'out_hg')


# In[4]:

#1x1卷积网络
def lin(inputs,numOut,is_training,scope=''):
    with slim.arg_scope([slim.conv2d],padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),

                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training, 'decay': 0.95}):
        return slim.conv2d(inputs,numOut,kernel_size=[1,1],scope=scope+'lin')


# In[9]:

class HgModel():
    def __init__(self,stages,joints):
        self.stages = stages
        self.stage_heatmap = []
        self.stage_loss = [0] * stages
        self.total_loss = 0
        self.input_image = None
        self.gt_heatmap = None
        self.learning_rate = 0
        self.merged_summary = None
        self.joints = joints
        self.batch_size = 16
    def build_model(self,input_image,iftrain):
        with tf.variable_scope('processing'):
            with slim.arg_scope([slim.conv2d],padding='SAME',
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),

                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': iftrain, 'decay': 0.95}):
#                 net=tf.pad(input_image,np.array([[0,0],[3,3],[3,3],[0,0]]))
                net=slim.conv2d(input_image,64,kernel_size=[7,7],stride=2,scope='conv1')
                net=residual(net,128,iftrain,scope='conv2')
                net=slim.max_pool2d(net,kernel_size=[2,2],stride=2,padding='SAME',scope='pool1')
                net=residual(net,128,iftrain,scope='conv3')
                inter=residual(net,256,iftrain,scope='conv4')
        out=[]
        for i in range(self.stages):
             with tf.variable_scope('stage_'+str(i+1)):
                    hg=hourglass(inter,4,256,iftrain,name='hg')
                    l1=hg
                    l1=residual(l1,256,iftrain,scope='l1')
                    l1=lin(l1,256,iftrain,scope='l2')
                    tmpout=slim.conv2d(l1,self.joints,kernel_size=[1,1],activation_fn=None,scope='l3')
                    out.append(tmpout)
                    if i< self.stages:
                        l1_=slim.conv2d(l1,256,kernel_size=[1,1],activation_fn=None,scope='l4')
                        tmpout_=slim.conv2d(tmpout,256,kernel_size=[1,1],activation_fn=None,scope='l5')
                        inter=tf.add_n([inter,l1_,tmpout_])
        self.stage_heatmap=out
                    
                    
                
    def build_loss(self, gt_heatmap, lr, lr_decay_rate, lr_decay_step,optimizer='Adam'):
        self.gt_heatmap = gt_heatmap
        self.total_loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.optimizer=optimizer

        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage+1) + '_loss'):
                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap[:,:,:,:13],
                                                       name='l2_loss') / self.batch_size
            tf.summary.scalar('stage' + str(stage+1) + '_loss', self.stage_loss[stage])

        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total_loss', self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.contrib.framework.get_or_create_global_step()

            self.cur_lr = tf.train.exponential_decay(self.learning_rate,
                                                 global_step=self.global_step,
                                                 decay_rate=self.lr_decay_rate,
                                                 decay_steps=self.lr_decay_step)
            tf.summary.scalar('learning_rate', self.cur_lr)
            
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.cur_lr)
            self.train_step = slim.learning.create_train_op(self.total_loss, self.optimizer, global_step=self.global_step)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                updates = tf.group(*update_ops)
                self.total_loss = control_flow_ops.with_dependencies([updates], self.total_loss)
        self.merged_summary = tf.summary.merge_all()
         
                    
                


# In[6]:

def print_current_training_stats(global_step, cur_lr, total_loss,total_loss1, time_elapsed):
    stats = 'Step: {}/{} ----- Cur_lr: {:1.7f} ----- Time: {:>2.2f} sec.'.format(global_step, 300000,
                                                                                 cur_lr, time_elapsed)
    print(stats)
    print('Training total_loss: {:>7.2f}  Testing total_loss:{:>7.2f}'.format(total_loss,total_loss1))


# In[7]:


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer(filename)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.string),
                                               'img_raw' : tf.FixedLenFeature([], tf.string),
                                           })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.decode_raw(features['label'], tf.float64)
    label = tf.reshape(label, [64, 64, 22])
    return img,label


# In[8]:

inpath='TC/tfrecord/glass/'
files = [f for f in os.listdir(inpath) ]
file_name=[inpath+fs for fs in files]
img, label = read_and_decode(file_name)

#使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=16, capacity=1000,
                                                min_after_dequeue=500)

img_test,label_test=read_and_decode(["TC/tfrecord/val/glass.tfrecords"])
img_tbatch, label_tbatch = tf.train.shuffle_batch([img_test, label_test],
                                                batch_size=16, capacity=1000,
                                                min_after_dequeue=500)


# In[ ]:


x=tf.placeholder(tf.float32,[None,256,256,3])
y_=tf.placeholder(tf.float32,[None,64,64,22])
phase = tf.placeholder(tf.bool, name='phase')
model=HgModel(3,21)


# In[ ]:


model.build_model(x,phase)
model.build_loss(y_,0.001,0.5,10000,optimizer='RMSProp')
merged_summary=model.merged_summary
min_loss=10000


# In[ ]:

with tf.Session() as sess:
    train_writer=tf.summary.FileWriter('TC/Graph/glass/train/',sess.graph)
    test_writer = tf.summary.FileWriter('TC/Graph/glass/test/', sess.graph)
    saver=tf.train.Saver(max_to_keep=3)
    
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    threads = tf.train.start_queue_runners(sess=sess)
    for training_itr in range(300000):
        
        batch_x_np, batch_gt_heatmap_np = sess.run([img_batch,label_batch])
        _, summaries, current_lr,stage_heatmap_np, global_step = sess.run([model.train_step,
                                                      merged_summary,
                                                      model.cur_lr,
                                                      model.stage_heatmap,
                                                      model.global_step
                                                      ],feed_dict={x: batch_x_np,y_: batch_gt_heatmap_np,phase:True})
        
        total_loss_np=sess.run(model.total_loss,feed_dict={x: batch_x_np,y_: batch_gt_heatmap_np,phase:True})
        t1=time.time()
        sess.run(model.stage_heatmap,feed_dict={x: batch_x_np,phase:False})
        t2=time.time()-t1
        train_writer.add_summary(summaries, global_step)
        if training_itr %5==0:
#             saver.save(sess=sess, save_path='model/hand_landmark_v6.1_model/model.ckpt',global_step=(global_step + 1))
            mean_val_loss = 0
            cnt = 0
            while cnt < 30:
                x_test,y_test=sess.run([img_tbatch,label_tbatch])
                total_loss_np1, summaries1 = sess.run([model.total_loss, merged_summary],
                                                        feed_dict={x: x_test,y_:y_test,phase:False})
                mean_val_loss += total_loss_np1
                cnt += 1
            print_current_training_stats(global_step, current_lr, total_loss_np,mean_val_loss / cnt, t2)
            test_writer.add_summary(summaries1, global_step)
            if mean_val_loss / cnt < min_loss:
                min_loss=mean_val_loss/cnt
                saver.save(sess=sess, save_path='TC/model/glass/model.ckpt',global_step=(global_step + 1))



