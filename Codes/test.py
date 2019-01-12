
import tensorflow as tf
from tensorflow.python.client import device_lib
#from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib import rnn
import numpy as np
import tensorflow.contrib.slim as slim

# to save the data in mat file
import scipy.io as sio
import os
import h5py
import math

from hilbert import hilbertCurve
import skimage.io as io
from matplotlib import pyplot as plt
import scipy.misc 


#tf.reset_default_graph()
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
log_device_placement = True
# Parameters
lr = 0.00003
training_iters = 50000000
batch_size = 8
display_step = 10
nb_nontamp_img=16960
nb_tamp_img=68355
nbFilter=32


# LSTM network parameters
n_steps = 64 # timesteps
nBlock=int(math.sqrt(n_steps))
n_hidden = 64# hidden layer num of features
nStride=int(math.sqrt(n_hidden))
# other parameters
imSize=256
# Network Parameters
n_classes = 2 # manipulated vs unmanipulated


# tf Graph input
input_layer = tf.placeholder("float", [None, imSize,imSize,3])
y= tf.placeholder("float", [2,None, imSize,imSize])
freqFeat=tf.placeholder("float", [None, 64,240])
ratio=15.0 #tf.placeholder("float",[1])
#out_rnn=tf.placeholder("float", [None, 128,128,3])



############################################################################
#total_layers = 25 #Specify how deep we want our network
units_between_stride = 2
upsample_factor=16
n_classes=2
beta=.01
outSize=16
############################################################################
seq = np.linspace(0,63,64).astype(int)
order3 = hilbertCurve(3)
order3 = np.reshape(order3,(64))
hilbert_ind = np.lexsort((seq,order3))
actual_ind=np.lexsort((seq,hilbert_ind))

weights = {
    'out': tf.Variable(tf.random_normal([64,64,nbFilter]))
}
biases = {
    'out': tf.Variable(tf.random_normal([nbFilter]))
}




with tf.device('/gpu:1'):

    def conv_mask_gt(z): 
        # Get ones for each class instead of a number -- we need that
        # for cross-entropy loss later on. Sometimes the groundtruth
        # masks have values other than 1 and 0. 
        class_labels_tensor = (z==1)
        background_labels_tensor = (z==0)

        # Convert the boolean values into floats -- so that
        # computations in cross-entropy loss is correct
        bit_mask_class = np.float32(class_labels_tensor)
        bit_mask_background = np.float32(background_labels_tensor)
        combined_mask=[]
        combined_mask.append(bit_mask_background)
        combined_mask.append(bit_mask_class)
        #combined_mask = tf.concat(concat_dim=3, values=[bit_mask_background,bit_mask_class])		

        # Lets reshape our input so that it becomes suitable for 
        # tf.softmax_cross_entropy_with_logits with [batch_size, num_classes]
        #flat_labels = tf.reshape(tensor=combined_mask, shape=(-1, 2))	
        return combined_mask#flat_labels

    def get_kernel_size(factor):
        #Find the kernel size given the desired factor of upsampling.
        return 2 * factor - factor % 2

    def upsample_filt(size):
        """
        Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
        """
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * \
            (1 - abs(og[1] - center) / factor)

    def bilinear_upsample_weights(factor, number_of_classes):
        """
        Create weights matrix for transposed convolution with bilinear filter
        initialization.
        """    
        filter_size = get_kernel_size(factor)

        weights = np.zeros((filter_size,filter_size,number_of_classes,number_of_classes), dtype=np.float32)    
        upsample_kernel = upsample_filt(filter_size)    
        for i in xrange(number_of_classes):        
            weights[:, :, i, i] = upsample_kernel    
        return weights


    def resUnit(input_layer,i,nbF):
        with tf.variable_scope("res_unit"+str(i)):
        #input_layer=tf.reshape(input_layer,[-1,64,64,3])
            part1 = slim.batch_norm(input_layer,activation_fn=None)
            part2 = tf.nn.relu(part1)
            part3 = slim.conv2d(part2,nbF,[3,3],activation_fn=None)
            part4 = slim.batch_norm(part3,activation_fn=None)
            part5 = tf.nn.relu(part4)
            part6 = slim.conv2d(part5,nbF,[3,3],activation_fn=None)	
            output = input_layer + part6
        return output

    #tf.reset_default_graph()

    def segNet(input_layer,bSize,freqFeat,weights,biases):
        # layer1: resblock, input size(256,256)
        layer1 = slim.conv2d(input_layer,nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))	
        layer1 =resUnit(layer1,1,nbFilter)
        layer1 = tf.nn.relu(layer1)
        layer2=slim.max_pool2d(layer1, [2, 2], scope='pool_'+str(1))
        # layer2: resblock, input size(128,128)   
        layer2 = slim.conv2d(layer2,2*nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(1))
        layer2 =resUnit(layer2,2,2*nbFilter)
        layer2 = tf.nn.relu(layer2)
        layer3=slim.max_pool2d(layer2, [2, 2], scope='pool_'+str(2))
        # layer3: resblock, input size(64,64) 
        layer3 = slim.conv2d(layer3,4*nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(2))
        layer3 =resUnit(layer3,3,4*nbFilter)
        layer3 = tf.nn.relu(layer3)
        layer4=slim.max_pool2d(layer3, [2, 2], scope='pool_'+str(3))
        # layer4: resblock, input size(32,32) 
        layer4 = slim.conv2d(layer4,8*nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(3))
        layer4 =resUnit(layer4,4,8*nbFilter)
        layer4 = tf.nn.relu(layer4)
        layer4=slim.max_pool2d(layer4, [2, 2], scope='pool_'+str(4))
        # end of layer4: resblock, input size(16,16)

        # lstm network 
        patches=tf.transpose(freqFeat,[1,0,2])
        patches=tf.gather(patches,hilbert_ind)
        patches=tf.transpose(patches,[1,0,2])         
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        xCell=tf.unstack(patches, n_steps, 1)
        # 2 stacked layers
        stacked_lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden),output_keep_prob=0.9) for _ in range(2)] )
        out, state = rnn.static_rnn(stacked_lstm_cell, xCell, dtype=tf.float32)
        # organizing the lstm output
        out=tf.gather(out,actual_ind)
        # convert to lstm output (64,batchSize,nbFilter)
        lstm_out=tf.matmul(out,weights['out'])+biases['out']
        lstm_out=tf.transpose(lstm_out,[1,0,2])
        # convert to size(batchSize, 8,8, nbFilter)
        lstm_out=tf.reshape(lstm_out,[bSize,8,8,nbFilter])
        # perform batch normalization and activiation
        lstm_out=slim.batch_norm(lstm_out,activation_fn=None)
        lstm_out=tf.nn.relu(lstm_out)
        # upsample lstm output to (batchSize, 16,16, nbFilter)
        temp=tf.random_normal([bSize,outSize,outSize,nbFilter])
        uShape1=tf.shape(temp)
        upsample_filter_np = bilinear_upsample_weights(2, nbFilter)
        upsample_filter_tensor = tf.constant(upsample_filter_np)
        lstm_out = tf.nn.conv2d_transpose(lstm_out, upsample_filter_tensor,output_shape=uShape1,strides=[1, 2, 2, 1])

        # reduce the filter size to nbFilter for layer4
        top = slim.conv2d(layer4,nbFilter,[1,1], normalizer_fn=slim.batch_norm, activation_fn=None, scope='conv_top')
        top = tf.nn.relu(top)
        # concatenate both lstm features and image features
        joint_out=tf.concat([top,lstm_out],3)
        # perform upsampling (batchSize, 64,64, 2*nbFilter)
        temp=tf.random_normal([bSize,outSize*4,outSize*4,2*nbFilter])
        uShape1=tf.shape(temp)
        upsample_filter_np = bilinear_upsample_weights(4, 2*nbFilter)
        upsample_filter_tensor = tf.constant(upsample_filter_np)
        upsampled_layer4 = tf.nn.conv2d_transpose(joint_out, upsample_filter_tensor,output_shape=uShape1,strides=[1, 4, 4, 1]) 	
        # reduce filter sizes	
        upsampled_layer4 = slim.conv2d(upsampled_layer4,2,[1,1], normalizer_fn=slim.batch_norm, activation_fn=None, scope='conv_'+str(4))
        upsampled_layer4=slim.batch_norm(upsampled_layer4,activation_fn=None)
        upsampled_layer4=tf.nn.relu(upsampled_layer4)
        # upsampling to (batchSize, 256,256, nbClasses)
        temp=tf.random_normal([bSize,outSize*16,outSize*16,2])
        uShape1=tf.shape(temp)
        upsample_filter_np = bilinear_upsample_weights(4,2)
        upsample_filter_tensor = tf.constant(upsample_filter_np)
        upsampled_layer5 = tf.nn.conv2d_transpose(upsampled_layer4, upsample_filter_tensor,output_shape=uShape1,strides=[1, 4, 4, 1]) 
        #upsampled_layer5=slim.batch_norm(upsampled_layer5,activation_fn=None)
        #upsampled_layer5 = slim.conv2d(upsampled_layer5,2,[3,3], normalizer_fn=slim.batch_norm, activation_fn=None, scope='conv_'+str(5))
        #upsampled_layer5=tf.nn.relu(upsampled_layer5)


        return upsampled_layer5


    y1=tf.transpose(y,[1,2,3,0])
    upsampled_logits=segNet(input_layer,batch_size,freqFeat,weights,biases)


    flat_pred=tf.reshape(upsampled_logits,(-1,n_classes))
    flat_y=tf.reshape(y1,(-1,n_classes))

    #loss1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_pred,labels=flat_y))

    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(flat_y,flat_pred, 1.0))

    trainer = tf.train.AdamOptimizer(learning_rate=lr)
    update = trainer.minimize(loss)
    #update2 = trainer.minimize(loss2)

    probabilities=tf.nn.softmax(upsampled_logits)
    correct_pred=tf.equal(tf.argmax(probabilities,1),tf.argmax(flat_y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    y_actual=tf.argmax(flat_y,1)
    y_pred=tf.argmax(flat_pred,1)

    #mask_actual= tf.argmax(y1,3)
    mask_pred=tf.argmax(upsampled_logits,3)

    #mask_actual= tf.argmax(y1,3)
    mask_p=tf.argmax(flat_pred,dimension=1)
    mask_pred=tf.reshape(mask_p,(-1,256,256))
    mask_act=tf.argmax(flat_y,dimension=1)
    mask_actual=tf.reshape(mask_act,(-1,256,256))

# Initializing the variables
init = tf.initialize_all_variables()
saver = tf.train.Saver()

config=tf.ConfigProto()
config.allow_soft_placement=True
config.log_device_placement=True
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4

with tf.Session(config=config) as sess:
    sess.run(init) 
    saver.restore(sess,'../model/final_model_nist.ckpt')	
    print 'session starting .................!!!!' 
    subtract_mean=True

    # loading NC16 data
    mx=127.0
    feat4=h5py.File('../test_data/test_imgs_feat_v2.hdf5','r')
    freq4=np.array(feat4["feat"])
    feat4.close()

    hdf5_file=h5py.File('../test_data/test_imgs_v2.hdf5','r')

    tx=np.array(hdf5_file['test_img'])
    tx=np.float32(tx)
    tx= np.multiply(tx,1.0/mx) 
    ty=np.array(hdf5_file['test_labels'])
    hdf5_file.close()
    # ====== #
    nTx=np.zeros((batch_size,256,256,3))
    nTy=np.zeros((batch_size,256,256))
    nTx1=np.zeros((batch_size,64,240))
    n1=0;n2=8
    for imNb in range(n1,n2):
        nTx[imNb-n1]=tx[imNb] 
        nTy[imNb-n1]=ty[imNb]
        nTx1[imNb-n1]=freq4[imNb]
    print np.shape(nTx)
    print np.shape(nTy)
    print np.shape(nTx1)
    ty_prime=conv_mask_gt(nTy)
    final_predictions, final_probabilities,y2=sess.run([mask_pred,probabilities,mask_actual], feed_dict={input_layer: nTx, y:ty_prime, freqFeat: nTx1})
    print np.shape(final_predictions)
    print np.shape(final_probabilities)

    #sio.savemat('pred_res.mat',{'img':nTx,'labels':nTy,'pred':final_predictions,'prob':final_probabilities,'gT':y2})
    nb = 0
    for i in range(n1,n2):
        print(i)
        cmap = plt.get_cmap('bwr')
        f,(ax,ax1,ax2,ax3)=plt.subplots(1,4,sharey=True)
        ax.imshow(tx[i])
        ax1.imshow(ty[i])
        ax2.imshow(final_predictions[nb])
        #ax3.set_title('Final Argmax')
        probability_graph = ax3.imshow(final_probabilities.squeeze()[nb, :,:, 0])
        nb += 1
        #ax3.set_title('Final Probability of the Class')
        plt.colorbar(probability_graph)
        plt.show()











 
