"""
Author: Jason Bunk

Build a classifier for precomputed features.
"""
import os,sys
thispath = os.path.dirname(os.path.abspath(__file__))

import time, gc
import numpy as np
import numpy.random as npr
import cv2
import cPickle
import tables
from utils import *
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def my_quadratic_log_probs(clf, X):
    assert len(X.shape) == 2
    values = clf._decision_function(X)
    valmax = np.amax(values, axis=1, keepdims=True)
    log_softmax_denom = valmax + np.log(np.sum(np.exp(values - valmax), axis=1,keepdims=True))
    return values - log_softmax_denom

def check_accuracy(preds, Y, desc='', descfile=None):
    checkme = np.argmax(preds, axis=1)
    assert checkme.shape == Y.shape, str(checkme.shape)+" vs "+str(Y.shape)
    assert np.amin(checkme) >= 0 and np.amax(checkme) <= 1
    assert np.amin(Y) >= 0 and np.amax(Y) <= 1
    eqs = np.equal(Y, checkme).astype(np.int64)
    numeq = float(np.sum(eqs))
    accurac = 100.0 * numeq / float(eqs.size)
    outstr = desc+"accuracy: "+str(accurac)+" %"
    if descfile is None:
        print(outstr)
    else:
        descfile.write(outstr+'\n')
    return accurac

def load_hdf5_train_set(filename, fraction_manip=0.50, flatten=True):
    fraction_manip = float(fraction_manip)

    h5file = tables.open_file(filename,mode='r')
    f0 = h5file.root.f0[:]
    f1 = h5file.root.f1[:]
    h5file.close()

    f0shape = f0.shape
    print("f0.shape "+str(f0.shape)+", dtype "+str(f0.dtype)+"; f1.shape "+str(f1.shape)+", dtype "+str(f1.dtype))

    if np.fabs(fraction_manip - 0.5) < 1e-9:
        assert f0.shape[1:] == f1.shape[1:], str(f0.shape)+" vs "+str(f1.shape)
        assert abs(int(f0.shape[0]) - int(f1.shape[0])) <= 1, str(f0.shape)+" vs "+str(f1.shape)
    else:
        print("------------- shuffling and splitting data for "+str(fraction_manip)+" fraction manip")
        npr.shuffle(f1) # shuffle so first N patches don't only come from first N/M images (M patches per image)
        num1 = int(np.ceil(float(f0.shape[0])*fraction_manip / (1. - fraction_manip)))
        assert num1 <= f1.shape[0], str(num1)+" vs "+str(f1.shape)
        f1 = f1[:num1, ...]
        print("f1.shape "+str(f1.shape)+", dtype "+str(f1.dtype))

    tr_X = np.concatenate((f0, f1), axis=0).copy() #.reshape((f0.shape[0]+f1.shape[0], -1)).copy()
    del f0
    del f1
    gc.collect()

    print("tr_X.shape "+str(tr_X.shape)+", dtype "+str(tr_X.dtype))

    tr_Y = np.ones((tr_X.shape[0],), dtype=np.int64)
    tr_Y[:int(f0shape[0])] = 0

    print("tr_Y.shape "+str(tr_Y.shape)+", dtype "+str(tr_Y.dtype))
    print("np.mean(tr_Y) == "+str(np.mean(tr_Y)))

    tr_X, tr_Y = shuffle_in_unison(tr_X, tr_Y)

    if flatten:
        tr_X = tr_X.reshape((tr_X.shape[0], -1))

    return tr_X.astype(np.float32).copy(), tr_Y.copy()

def onehot(YY):
    if len(YY.shape) == 1:
        #describe("YY",YY)
        YY = np.pad(np.expand_dims(YY,-1), ((0,0),(1,0)), mode='constant').astype(np.float32)
        #describe("YY",YY)
        assert np.amax(np.fabs(YY[:,0])) < 1e-9
        YY[:,0] = 1. - YY[:,1]
    return YY

if __name__ == '__main__':
    try:
        traindatafilename = sys.argv[1]
    except:
        print("usage:   {train-set-filename}   {optional:valid-set-filename}   {optional:scores-out-filename}  {optional:plot-title}")
        quit()
    validfilename = None
    scoresfname = None
    if len(sys.argv) > 2:
        validfilename = sys.argv[2]
    if len(sys.argv) > 3:
        scoresfname = sys.argv[3]
    plottitle = None
    if len(sys.argv) > 4:
        plottitle = sys.argv[4]

    while traindatafilename.endswith('.'):
        traindatafilename = traindatafilename[:-1]
    while validfilename.endswith('.'):
        validfilename = validfilename[:-1]

    COMPUTE_COVARIANCES = False

    tr_X, tr_Y = load_hdf5_train_set(traindatafilename+'.hdf5')
    va_X, va_Y = load_hdf5_train_set(validfilename+'.hdf5')

    if False:
        #===========================================================================
        # shuffle train/val datasets,
        # so that we get a different train/val set each time
        # we can then run this multiple times to get a cross-validated ensemble
        #tvall_X, tvall_Y = shuffle_in_unison(np.concatenate((tr_X, va_X), axis=0), \
        #                                     np.concatenate((tr_Y, va_Y), axis=0))
        #totalnumdata = tvall_X.shape[0]
        #assert totalnumdata == tvall_Y.shape[0]
        #nval = int(round(  0.15*float(totalnumdata)  ))

        #va_X = tvall_X[:nval,:]
        #va_Y = tvall_Y[:nval]

        #tr_X = tvall_X[nval:,:]
        #tr_Y = tvall_Y[nval:]

        #del tvall_X
        #del tvall_Y
        #gc.collect()
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ shuffled train/val datasets")
        #print("shuffled: va_Y[:10] == "+str(va_Y[:10]))
        #===========================================================================

    describe("BEFORE MEAN SUBTR: tr_X", tr_X)

    tr_mean_vec = np.mean(tr_X, axis=0, keepdims=True)
    tr_X -= tr_mean_vec
    tr_scale_vec = np.std(tr_X, axis=0, keepdims=True)
    tr_X /= tr_scale_vec
    #print("BEFORE: tr_X.shape "+str(tr_X.shape)+", dtype "+str(tr_X.dtype))
    describe("AFTER MEAN SUBTR: tr_X", tr_X)
    #assert np.amax(np.fabs(np.mean(tr_X,axis=0))) < 1e-4, str(np.mean(tr_X,axis=0))
    #assert len(tr_X.shape) == 2

    va_X -= tr_mean_vec
    va_X /= tr_scale_vec
    describe("AFTER MEAN SUBTR: va_X", va_X)

    beftime = time.time()

    assert scoresfname is not None
    with open(scoresfname, 'w') as scoresoutfile:

        if True:
            print("tr_Y[:10]")
            print(str(tr_Y[:10]))
            #print("\n")

            tr_Y = onehot(tr_Y)
            va_Y = onehot(va_Y)

            print("tr_Y[:10]")
            print(str(tr_Y[:10].transpose()))
            #print("\n")

            #############################################################
            import tensorflow as tf
            from keras.backend.tensorflow_backend import set_session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            #config.gpu_options.per_process_gpu_memory_fraction = 0.3
            set_session(tf.Session(config=config))
            #############################################################
            import keras
            from keras.models import Sequential
            from keras.layers import Dense, Dropout, Activation
            from keras.regularizers import l2 as keras_l2

            class KerasEpochEndCallbackRocAUC(keras.callbacks.Callback):
                def __init__(self):
                    super(keras.callbacks.Callback, self).__init__()
                def on_epoch_end(self, epoch, logs={}):
                    y_pred = self.model.predict_proba(va_X, verbose=0)
                    score = roc_auc_score(va_Y, y_pred)
                    print("\nepoch {:d} - ROC AUC score: {:.6f}".format(epoch, score))

            bestmodels = []
            for droprate in [0.16, 0.22, 0.28, 0.35]:
                for l2regdenom in [3000., 5000., 8000., 12000.]:
                    parmdescripstr = "droprate: "+str(droprate)+", l2regdenom: "+str(l2regdenom)

                    BATCH_SIZE = 1024
                    L2REG = 1. / l2regdenom
                    model = Sequential()
                    model.add(Dense(256, input_dim=int(tr_X.shape[1]), init='glorot_normal', activation='elu', W_regularizer=keras_l2(L2REG)))
                    model.add(Dropout(droprate))
                    model.add(Dense(256, init='glorot_normal', activation='elu', W_regularizer=keras_l2(L2REG)))
                    model.add(Dropout(droprate))
                    model.add(Dense(2, init='glorot_normal', b_regularizer=keras_l2(0.04), W_regularizer=keras_l2(L2REG)))
                    model.add(Activation('softmax'))
                    model.compile(loss='categorical_crossentropy',
                                    optimizer=keras.optimizers.Adam(epsilon=1e-4),
                                    metrics=['accuracy',])
                    model.fit(tr_X, tr_Y,
                                nb_epoch=25,
                                batch_size=BATCH_SIZE, callbacks=[KerasEpochEndCallbackRocAUC(),])

                    ypred = model.predict_proba(va_X, verbose=0)
                    score = roc_auc_score(va_Y, ypred)
                    bestmodels.append((score, model, parmdescripstr))
                    print("------ trained model ROC AUC score: "+str(score))

            bestmodels = sorted(bestmodels, key=lambda x:x[0])
            score,model,parmdescripstr = bestmodels[-1]
            print("best model\'s ROC AUC score: "+str(score)+", "+parmdescripstr)

            model.save(scoresfname+'_KerasResult.h5')
            h5file = tables.open_file(scoresfname+'_KerasResult2.h5',mode='w')
            h5file.create_array(h5file.root, 'tr_mean_vec', tr_mean_vec)
            h5file.create_array(h5file.root, 'tr_scale_vec', tr_scale_vec)
            h5file.close()

            preds = model.predict(va_X)
            describe("preds", preds)

            fpr, tpr, thresholds = roc_curve(va_Y[:,1], preds[:,1])
            roc_auc = auc(fpr, tpr)

            highestroc = (1, roc_auc, fpr, tpr, thresholds)

        else:
            clf = QuadraticDiscriminantAnalysis(store_covariances=COMPUTE_COVARIANCES)
            clf.fit(tr_X, tr_Y)

            afttime = time.time()
            print("AFTER: tr_X.shape "+str(tr_X.shape)+", dtype "+str(tr_X.dtype))
            print("fitting took "+str(afttime - beftime)+" sec")

            fraction_manip = 0.50

            # delete training set, since it is no longer needed
            del tr_X
            del tr_Y
            gc.collect()

            clf.means_ = np.array(clf.means_)
            clf.priors_ = np.array([1.-fraction_manip, fraction_manip]) #
            clf.rotations_ = np.array(clf.rotations_)
            clf.scalings_ = np.array(clf.scalings_)

            print("clf.priors_ == "+str(clf.priors_))

            # re-load validation set
            tr_X, tr_Y = load_hdf5_train_set(validfilename+'.hdf5', fraction_manip=fraction_manip)

            describe("VALIDATION: BEFORE MEAN SUBTR: tr_X", tr_X)
            #tr_X -= tr_mean_vec
            #tr_X /= tr_scale_vec
            #describe("VALIDATION: AFTER MEAN SUBTR: tr_X", tr_X)

            h5file = tables.open_file(scoresfname+'_LDAresult.hdf5',mode='w')
            if COMPUTE_COVARIANCES:
                h5file.create_array(h5file.root, 'covariances', clf.covariances_)
            h5file.create_array(h5file.root, 'means', clf.means_)
            h5file.create_array(h5file.root, 'priors', clf.priors_)
            h5file.create_array(h5file.root, 'rotations', clf.rotations_)
            h5file.create_array(h5file.root, 'scalings', clf.scalings_)
            h5file.close()

            origscalings = clf.scalings_.copy()
            aaaaaa = np.linspace(-8.0, -0.05, num=100)
            isfirstiter = True
            highestroc = None
            for aaa in list(aaaaaa):
                if isfirstiter:
                    REGPARAM = 0.
                else:
                    REGPARAM = np.exp(aaa)
                    clf.scalings_ = ((1. - REGPARAM) * origscalings) + REGPARAM
                isfirstiter = False

                lognewprobs = my_quadratic_log_probs(clf, tr_X)

                oneprobs = np.exp(lognewprobs[:,1])
                fpr, tpr, thresholds = roc_curve(tr_Y, oneprobs)
                roc_auc = auc(fpr, tpr)
                distsfromdiag = (tpr - fpr) / np.sqrt(2.)
                maxfromdiag_idx = np.argmax(distsfromdiag)
                maxfromdiag_val = distsfromdiag[maxfromdiag_idx]
                assert np.fabs(maxfromdiag_val - np.amax(distsfromdiag)) < 1e-12

                if highestroc is None:
                    highestroc = (REGPARAM, roc_auc, fpr, tpr, thresholds)
                else:
                    if roc_auc > highestroc[1]:
                        highestroc = (REGPARAM, roc_auc, fpr, tpr, thresholds)

                check_accuracy(lognewprobs, tr_Y, desc= \
                    'REGPARAM '+str(REGPARAM)+ \
                    ', ROC AUC '+str(roc_auc)+ \
                    ', maxfromdiag '+str(maxfromdiag_val)+ \
                    ', at fpr '+str(fpr[maxfromdiag_idx])+ \
                    ', tpr '+str(tpr[maxfromdiag_idx])+ \
                    ', threshold '+str(thresholds[maxfromdiag_idx])+ \
                    ', ', descfile=scoresoutfile)

        cPickle.dump(highestroc, open(scoresfname+'_LDAhighestROC.pkl','wb'))
        print("       (saved to \'"+scoresfname+"\')")
        print("highest AUC: "+str(highestroc[1])+" from regularization "+str(highestroc[0]))

        if True:
            idealdistsfromdiag = (1. - highestroc[2]) / np.sqrt(2.)
            distsfromdiag = (highestroc[3] - highestroc[2]) / np.sqrt(2.)
            maxfromdiag_idx = np.argmax(distsfromdiag)
            maxfromdiag_val = distsfromdiag[maxfromdiag_idx]
            maxfromdiag_string = 'maxfromdiag at fpr '+fstr4(highestroc[2][maxfromdiag_idx])+', tpr '+fstr4(highestroc[3][maxfromdiag_idx])
            print(maxfromdiag_string)

            lw = 2
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.plot(highestroc[2], highestroc[3], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % highestroc[1])
            plt.plot(highestroc[2], distsfromdiag, color='green', label='dists from diag')
            plt.plot(highestroc[2], idealdistsfromdiag, color='green', linestyle='--', label='ideal dists from diag')
            titlestr = 'Classifying patches'
            if plottitle is not None:
                titlestr += ', '+str(plottitle)
            titlestr += '\n' + 'best AUC: '+fstr4(highestroc[1]) +'\n'+maxfromdiag_string
            plt.title(titlestr)
            plt.xlabel('false positive rate')
            plt.ylabel('true positive rate')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.grid(True)
            plt.axes().set_aspect(1.)
            plt.subplots_adjust(top=0.85)
            #plt.show()
            plt.savefig(scoresfname+'.png')
