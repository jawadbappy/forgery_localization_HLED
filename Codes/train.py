
import tensorflow as tf
from tensorflow.python.client import device_lib
#from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib import rnn
import numpy as np
import tensorflow.contrib.slim as slim
from compute_mcc import *
#import scipy.io as sio
import os
import math
import h5py
#from compute_mcc import compute_mcc,metrics,_fast_hist,label_accuracy_score
from hilbert import hilbertCurve
#from compute_IoU import compute_precision,bb_IoU

#tf.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
log_device_placement = True
# Parameters
lr = 0.00003
training_iters = 50000000
batch_size = 16
display_step = 10
nb_nontamp_img=16960
nb_tamp_img=68355
nbFilter=32


# LSTM network parameters
n_input = 240 # data input (img shape: 64x64)
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

    #all_weights  = tf.trainable_variables()
    #regLoss = tf.add_n([ tf.nn.l2_loss(v) for v in all_weights ]) * beta
    #loss = 0.75*loss1+loss2
    trainer = tf.train.AdamOptimizer(learning_rate=lr)
    update = trainer.minimize(loss)
    #update2 = trainer.minimize(loss2)

    probabilities=tf.nn.softmax(flat_pred)
    correct_pred=tf.equal(tf.argmax(probabilities,1),tf.argmax(flat_y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    y_actual=tf.argmax(flat_y,1)
    y_pred=tf.argmax(flat_pred,1)

    mask_actual= tf.argmax(y1,3)
    mask_pred=tf.argmax(upsampled_logits,3)


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

    # loading data
    feat1=h5py.File('../data/train_data_feat_v2.hdf5','r')
    freq1=np.array(feat1["feat"])
    feat1.close()
    mx=127.0
    hdf5_file=h5py.File('../data/train_data.hdf5','r')
    Img=np.array(hdf5_file["train_img"])
    #Img=np.uint8(Img)
    #Img=np.multiply(Img,1.0/mx)
   
    Lab=np.array(hdf5_file["train_labels"])

    subtract_mean = True
    step = 1
    if subtract_mean:
        mm = hdf5_file["train_mean"][0, ...]
        mm = mm[np.newaxis, ...]
    hdf5_file.close()
    ## loading nist data
    feat2=h5py.File('../data/MFC18_data_feat_v2.hdf5','r')
    freq2=np.array(feat2["feat"])
    feat2.close()

    nist=h5py.File('../data/MFC18_data.hdf5','r')
    nI=np.array(nist["train_img"])
    #nI=np.uint8(nI)
    #nI=np.multiply(nI,1.0/mx)
    nb_nist_img=np.shape(nI)[0]-batch_size
    print np.shape(nI)
    nL=np.array(nist["train_labels"])
    print np.shape(nL)	

    subtract_mean = False
    step = 1
    if subtract_mean:
        mm1 = nist["train_mean"][0, ...]
        mm1 = mm1[np.newaxis, ...]
    nist.close()

    # Keep training until reach max iterations
    print 'epoch 1 sarted......'
    
    # loading nc_2017_eval_set 
    feat3=h5py.File('../data/eval_set_v3_feat_v2.hdf5','r')
    freq3=np.array(feat3["feat"])
    feat3.close()

    nc17=h5py.File('../data/eval_set_v3.hdf5','r')
    nc17_img=np.array(nc17["test_img"])
    #nc17_img=np.uint8(nc17_img)
    #nc17_img=np.multiply(nc17_img,1.0/mx)
    nb_nc17_img=np.shape(nc17_img)[0]
    print np.shape(nc17_img)
    nc17_lab=np.array(nc17["test_labels"])
    print np.shape(nc17_lab)

    subtract_mean = True
    step = 1
    if subtract_mean:
        mm3 = nc17["test_mean"][0, ...]
        mm3 = mm3[np.newaxis, ...]
    nc17.close()



    feat=h5py.File('../data/nc16_train_feat.hdf5','r')
    feat_nc16=np.array(feat["feat"])
    feat.close()

    hdf5_file=h5py.File('../data/nc16_FT_v1.hdf5','r')
    Img_NC16=np.array(hdf5_file["train_img"])
    print np.shape(Img_NC16)
    Lab_NC16=np.array(hdf5_file["train_labels"])
    print np.shape(Lab_NC16)
    nb_nc16_img=np.shape(Img_NC16)[0]
    tx=np.array(hdf5_file['test_img'])
    tx=np.float32(tx)
    tx= np.multiply(tx,1.0/mx) 
    ty=np.array(hdf5_file['test_labels'])
    hdf5_file.close()
    
     # validation set
    dx=nI[-batch_size:]
    #dx=np.float32(dx)
    dx=np.multiply(dx,1.0/mx)
    dx1=freq2[-batch_size:]
    #dx-=mm1
    dy=nL[-batch_size:]
    dy=conv_mask_gt(dy)
    # separate out the training set from val set
    nI=nI[:-batch_size]
    nL=nL[:-batch_size]
    print np.shape(nI)

    feat4=h5py.File('../data/nc16_test_feat.hdf5','r')
    freq4=np.array(feat4["feat"])
    feat4.close() 
    # finished data loading

    # Keep training until reach max iterations
    print 'epoch 1 sarted......'

    ## tunable parameters
    epoch_iter=0;
    iter_nontamp=0;iter_tamp=0;iter_nist=0;iter_nc17=0;iter_nc16=0
    
    epoch_iter_nontamp=int(nb_nontamp_img/2)
    epoch_iter_tamp=int(nb_tamp_img/6)
    epoch_iter_nist=int(nb_nist_img/2)
    epoch_iter_nc17=int(nb_nc17_img/2)
    epoch_iter_nc16=int(nb_nc16_img/6)
    
    #bUtamp=2;
    bTamp=6;bNist=2;bNc17=2;bnctamp=6

    best_acc=np.float32(0.45)
    best_prec=np.float32(0.2)
    best_acc1=np.float32(0.45)
    best_prec1=np.float32(0.15)
    
    batch_x=np.zeros((batch_size,imSize,imSize,3))
    batch_y=np.zeros((batch_size,imSize,imSize))
    batch_x1=np.zeros((batch_size,64,240))

    while step * batch_size < training_iters:
        if (iter_nc16 % epoch_iter_nc16)==0:
            #print "data loading for nc16 ..."
            iter_nc16=0		
            in_size=np.shape(Img_NC16)[0]
            arr_ind=np.arange(in_size)
            np.random.shuffle(arr_ind)
            im_nc16 = Img_NC16[arr_ind, ...]
            Y_nc16 = Lab_NC16[arr_ind, ...]
            fr_nc16=feat_nc16[arr_ind, ...]

            im_nc16=im_nc16[:(np.shape(arr_ind)[0]/bnctamp)*bnctamp,...]
            Y_nc16=Y_nc16[:(np.shape(arr_ind)[0]/bnctamp)*bnctamp,...]
            fr_nc16=fr_nc16[:(np.shape(arr_ind)[0]/bnctamp)*bnctamp,...]

        if (iter_tamp % epoch_iter_tamp)==0:
            print "data loading for synthesized images ..."
            iter_tamp=0
            in_size=nb_tamp_img
            arr_ind=np.arange(in_size)
            np.random.shuffle(arr_ind)
            arr_ind=arr_ind+nb_nontamp_img
            im2 = Img[arr_ind, ...]
            Y2 = Lab[arr_ind, ...]
            fr2=freq1[arr_ind, ...]
            epoch_iter+=1
            print "epoch finished..starting next epoch..>>>"

            im2=im2[:(np.shape(arr_ind)[0]/bTamp)*bTamp,...]
            Y2=Y2[:(np.shape(arr_ind)[0]/bTamp)*bTamp,...]
            fr2=fr2[:(np.shape(arr_ind)[0]/bTamp)*bTamp,...]

        if (iter_nist % epoch_iter_nist)==0:
            print "data loading for mfc18 images ..."
            iter_nist=0
            in_size=np.shape(nI)[0]
            arr_ind=np.arange(in_size)
            np.random.shuffle(arr_ind)
            im3 = nI[arr_ind, ...]
            Y3 = nL[arr_ind, ...]
            fr3=freq2[arr_ind, ...]

            im3=im3[:(np.shape(arr_ind)[0]/bNist)*bNist,...]
            Y3=Y3[:(np.shape(arr_ind)[0]/bNist)*bNist,...]
            fr3=fr3[:(np.shape(arr_ind)[0]/bNist)*bNist,...]

        if (iter_nc17 % epoch_iter_nc17)==0:
            print "data loading for nc17 eval images ..."
            iter_nc17=0
            in_size=np.shape(nc17_img)[0]
            arr_ind=np.arange(in_size)
            np.random.shuffle(arr_ind)
            im4 = nc17_img[arr_ind, ...]
            Y4 = nc17_lab[arr_ind, ...]
            fr4=freq3[arr_ind, ...]

            im4=im4[:(np.shape(arr_ind)[0]/bNc17)*bNc17,...]
            Y4=Y4[:(np.shape(arr_ind)[0]/bNc17)*bNc17,...]
            fr4=fr4[:(np.shape(arr_ind)[0]/bNc17)*bNc17,...]

        batch_x[:6,...]=np.float32(im_nc16[(iter_nc16*bnctamp): min((iter_nc16+1)*bnctamp, nb_nc16_img),...])
        batch_y[:6,...]=Y_nc16[(iter_nc16*bnctamp):min((iter_nc16+1)*bnctamp, nb_nc16_img),...]
        batch_x1[:6,...]=fr_nc16[(iter_nc16*bnctamp): min((iter_nc16+1)*bnctamp, nb_nc16_img),...]

        batch_x[6:12,...]=np.float32(im2[(iter_tamp*bTamp):min((iter_tamp+1)*bTamp,nb_tamp_img),...])
        batch_y[6:12,...]=Y2[(iter_tamp*bTamp):min((iter_tamp+1)*bTamp,nb_tamp_img),...]
        batch_x1[6:12,...]= fr2[(iter_tamp*bTamp):min((iter_tamp+1)*bTamp,nb_tamp_img),...]

        batch_x[12:14,...]=np.float32(im3[(iter_nist*bNist): min((iter_nist+1)*bNist, nb_nist_img),...])
        batch_y[12:14,...]=Y3[(iter_nist*bNist): min((iter_nist+1)*bNist, nb_nist_img),...]
        batch_x1[12:14,...]=fr3[(iter_nist*bNist): min((iter_nist+1)*bNist, nb_nist_img),...]

        batch_x[14:16,...]=np.float32(im4[(iter_nc17*bNc17): min((iter_nc17+1)*bNc17, nb_nc17_img),...])
        batch_y[14:16,...]=Y4[(iter_nc17*bNc17): min((iter_nc17+1)*bNc17, nb_nc17_img),...]
        batch_x1[14:16,...]=fr4[(iter_nc17*bNc17): min((iter_nc17+1)*bNc17, nb_nc17_img),...]

        ## data iterations
        iter_nontamp+=1;iter_tamp+=1;iter_nist+=1; iter_nc17+=1;iter_nc16+=1

        rev_batch_y=np.array(conv_mask_gt(batch_y))	
        if np.shape(batch_x)[0]!= batch_size:
            continue
        batch_x=np.multiply(batch_x,1.0/mx)
        sess.run(update, feed_dict={input_layer: batch_x, y: rev_batch_y, freqFeat: batch_x1})

        if step % display_step == 0:
            TP = 0; FP = 0;TN = 0; FN = 0
            # Calculate batch accuracy
         
            acc,cost,y1,p1= sess.run([accuracy,loss,y_actual,y_pred], feed_dict={input_layer: dx, y: dy, freqFeat: dx1})     
            # Calculate batch loss
            #cost = sess.run(, feed_dict={input_layer: dx, y: dy}) 
            
            a,b,c,d=compute_pos_neg(y1,p1)
            TP+=a; FP+=b;TN+=c; FN+=d
            prec=metrics(TP,FP,TN,FN)
            
            print "Iter " + str(step*batch_size) + ", Loss= " + str(cost) +  \
              ", epoch= " + str(epoch_iter)+ \
              ", batch= "+ str(iter_tamp) +  ", acc= "+ str(acc)+ ", precision= "+str(prec)
              

        if step % 100== 0:
            TP = 0; FP = 0;TN = 0; FN = 0 
            #TP1=0;FP1=0
            num_images=batch_size
            n_chunks=np.shape(tx)[0]/batch_size
            tAcc=np.zeros(n_chunks)

            for chunk in range(0,n_chunks):               
                tx_batch=tx[((chunk)*num_images):((chunk+1)*num_images),...]
                ty_batch=ty[((chunk)*num_images):((chunk+1)*num_images),...]
                tx1_batch=freq4[((chunk)*num_images):((chunk+1)*num_images),...]
                ty_batch=conv_mask_gt(ty_batch)
                tAcc[chunk],y2,p2=sess.run([accuracy,y_actual,y_pred], feed_dict={input_layer: tx_batch, y:ty_batch, freqFeat: tx1_batch})
            a,b,c,d=compute_pos_neg(y2,p2)

            TP+=a; FP+=b;TN+=c; FN+=d
            
            prec=metrics(TP,FP,TN,FN)            
            test_accuracy=np.mean(tAcc)
           
            if prec > best_acc :
                best_prec = prec
                save_path=saver.save(sess,'../model/final_model_nist.ckpt')
                print "Best Model Found on NC16..."
            
            print  "prec = "+str(prec)+"("+str(best_prec)+")" + ", acc = "+ str(test_accuracy)
            
        step += 1

        if step % 500 ==0: 
            save_path=saver.save(sess,'../model/final_model_nist.ckpt')
            print 'model saved ..........#epoch->'+str(epoch_iter)
    print "Optimization Finished!"


    
    
