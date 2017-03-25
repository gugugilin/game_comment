import tensorflow as tf
import numpy as np

class Poslect:
    def __init__(self):
        print("building...")
        self.build_graph()
        self.graph=True
        print("building done")
        return
    def _loop_body(self,time,keep_prob,x_image,ht_1,output,W_conv1_1,b_conv1_1,W_conv1_2,b_conv1_2,W_conv2,b_conv2,W_fc2,b_fc2,W_fc3,b_fc3,W_fc1,b_fc1):
        ht_image = tf.reshape(ht_1, [1,264, 264, 1])
        space_img=tf.concat([ht_image,[x_image[time]]],3)#output size 264x264x2
        #conv1_1
        
        h_conv1_1 = tf.nn.relu(self.conv2d(space_img, W_conv1_1) + b_conv1_1) # output size 264x264x16
        h_pool1_1 = self.max_pool_3x3(h_conv1_1) # output size 132x132x16
        
        #conv1_2
        
        h_conv1_2 = tf.nn.relu(self.conv2d([x_image[time]], W_conv1_2) + b_conv1_2) # output size 264x264x32
        h_pool1_2 = self.max_pool_3x3(h_conv1_2) # output size 132x132x32
        
        #with constrace of that streak and space 
        constrace=tf.concat([h_pool1_1,h_pool1_2],3)# output size 132x132x48
        
        #conv2
        
        h_conv2 = tf.nn.relu(self.conv2d(constrace, W_conv2) + b_conv2) # output size 132x132x64
        h_pool2 = self.max_pool_3x3(h_conv2) # output size 66x66x64
        
        #fc 1024
        
        h_pool2_flat = tf.reshape(h_pool2, [1, 66*66*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        #fc 40
        
        prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
        #fc center 40->264*264
        
        ht_2=tf.matmul(prediction, W_fc3) + b_fc3

        return [time+1,keep_prob,x_image,ht_2,prediction,W_conv1_1,b_conv1_1,W_conv1_2,b_conv1_2,W_conv2,b_conv2,W_fc2,b_fc2,W_fc3,b_fc3,W_fc1,b_fc1]
        
    def build_graph(self):
        self.xs = tf.placeholder(tf.float32, [None, 264*264],name='xs') 
        self.ys = tf.placeholder(tf.float32, [None, 40],name='ys')
        self.keep_prob = tf.placeholder(tf.float32)
        self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')
        x_image = tf.divide(tf.reshape(self.xs, [-1, 264, 264, 1]),256)
        self.ht_1=tf.placeholder(tf.float32, [1, 264*264]) 
        time = tf.constant(0, dtype=tf.int32)
        output = tf.Variable(tf.zeros([1,40]), name="output")
        W_conv1_1 = self.weight_variable([5,5, 2,16]) # patch 5x5, in size 2, out size 16
        b_conv1_1 = self.bias_variable([16])
        W_conv1_2 = self.weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
        b_conv1_2 = self.bias_variable([32])
        W_conv2 = self.weight_variable([5,5, 48,64]) # patch 5x5, in size 48, out size 64
        b_conv2 = self.bias_variable([64],1)
        W_fc2 = self.weight_variable([1024, 40])
        b_fc2 = self.bias_variable([40])
        W_fc3 = self.weight_variable([40, 264*264])
        b_fc3 = self.bias_variable([264*264])
        W_fc1 = self.weight_variable([66*66*64, 1024])
        b_fc1 = self.bias_variable([1024])
        
        final_results = tf.while_loop(
            cond=lambda time, *_: time < self.sequence_length,
            body=self._loop_body,
            loop_vars=(time,self.keep_prob,x_image,self.ht_1,output,W_conv1_1,b_conv1_1,W_conv1_2,b_conv1_2,W_conv2,b_conv2,W_fc2,b_fc2,W_fc3,b_fc3,W_fc1,b_fc1)
        )
        
        self.prediction=final_results[4]
        self.loss=tf.reduce_mean(tf.square(self.ys-self.prediction))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        
        return 
        
    def fit(self,images,targets,observe_len,epoch,my_keep_prob=0.6,training=True):
        
        saver = tf.train.Saver()
        init=tf.initialize_all_variables()
        if self.graph is True:
            with tf.Session() as sess:
                sess.run(init)
                # Do some work with the model.
                count=1
                input_data=[]
                output_data=[]
                init_ht_1=np.zeros([1,264*264])
                for image in images:
                    input_data.append(image)
                    output_data.append(targets[count-1])
                    if count>=observe_len:
                        del input_data[0]
                        del output_data[0]
                        if training is True:
                            for i in range(epoch):
                                sess.run(self.train_step, feed_dict={self.xs: np.array(input_data), self.ys: output_data,self.ht_1:init_ht_1,self.keep_prob: my_keep_prob,self.sequence_length:observe_len})
                        else:
                            sess.run(self.prediction, feed_dict={xs: input_data, ys: output_data,ht_1:init_ht_1,keep_prob: 1,sequence_length:observe_len})
                        print(self.loss.eval())
                    else:
                        count+=1
                # Save the variables to disk.
                if training is True:
                    save_path = saver.save(sess, "./ckpt/Poslect.ckpt")
                    print("Model saved in file: ", save_path)
        return 
    
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial,name='w')

    def bias_variable(self,shape,count=0):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial,name='b'+str(count))

    def conv2d(self,x, W):
        print(len(x))
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_3x3(self,x):
        return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')