#!/usr/bin/env python
"""
Author: Jason Bunk

Iterate over some experiments, building a binary classifier for each one.
This forms a set of binary classifiers that characterize resampling.
"""
import os,sys
try:
    handcraftedfeats = int(sys.argv[1])
except:
    print("usage:  {handcrafted-feats?}")
    print("        if false, precomputed feats are simply RGB, and no classifier is trained")
    quit()

doublefftnorm = 1

UCID_path = '/data/jbunk/UCID_plus_some_RAISE'

#UCID_train = 'train_with_halfsizeraise_and_ucid.hdf5'
#UCID_val   = 'val_with_halfsizeraise_and_ucid.hdf5'
#UCID_train = 'train_patch192_newuncomp_2xraise_ucid.hdf5'
#UCID_val   = 'val_patch192_newuncomp_2xraise_ucid.hdf5'
#UCID_train = 'train_patch256_newuncomp_2xraise_ucid_ONEPER.hdf5'
#UCID_val   = 'val_patch256_newuncomp_2xraise_ucid_ONEPER.hdf5'
#UCID_train = 'raise_patch320_newuncomp_2xraise_ucid_ONEPER_NEW.hdf5'
#UCID_val   =  'ucid_patch320_newuncomp_2xraise_ucid_ONEPER_NEW.hdf5'

UCID_train = 'train_patch192_2xraise_ucid_apr24_2017.hdf5'
UCID_val   =   'val_patch192_2xraise_ucid_apr24_2017.hdf5'

#classifierpath = '/home/jbunk/code/SD19new_march2017'

baseline_transf  = '[0.35, 0, 0]'
baseline_jpgpre  = '[0.95, 65, 100]'
baseline_jpgpost = '[0.95, 65, 100]'
baseline_angles  = '[0.5, -90., 90.]'
baseline_shear   = '[0.2, -3.5, 3.5]'
baseline_rescal  = '[0.6, -0.693147181, 0.693147181]'

pr_keys = ['transf_pr', 'jpgpre_pr', 'jpgpost_pr', 'angles_pr', 'shear_pr', 'rescal_pr']
pr_key_to_baseline = {'transf_pr': baseline_transf,   \
                      'jpgpre_pr': baseline_jpgpre,   \
                      'jpgpost_pr': baseline_jpgpost, \
                      'angles_pr': baseline_angles,   \
                      'shear_pr': baseline_shear,     \
                      'rescal_pr': baseline_rescal}

experims = [                                         \
                                                     \
{'title': 'jpg_lowqual',                             \
 'BAS_jpgpre_pr':  '[0.95, 85, 100]',                \
 'BAS_jpgpost_pr': '[0.95, 85, 100]',                \
 'MOD_jpgpre_pr': '[1.,   50,  85]'},                \
                                                     \
{'title': 'rotation_neg',                            \
 'BAS_angles_pr': '[0.5,  1.,  90.]',                \
 'MOD_transf_pr': '[1.,  0, 0]',                     \
 'MOD_angles_pr': '[1.,  -90., -1.]'},               \
                                                     \
{'title': 'rotation_pos',                            \
 'BAS_angles_pr': '[0.5, -90., -1.]',                \
 'MOD_transf_pr': '[1.,  0, 0]',                     \
 'MOD_angles_pr': '[1.,   1.,  90.]'},               \
                                                     \
{'title': 'rescale_down',                            \
 'BAS_rescal_pr': '[0.5, 0.01, 0.693147181]',        \
 'MOD_transf_pr': '[1.,  0, 0]',                     \
 'MOD_rescal_pr': '[1., -0.693147181, -0.01]'},      \
                                                     \
{'title': 'rescale_up',                              \
 'BAS_rescal_pr': '[0.5, -0.693147181, -0.01]',      \
 'MOD_transf_pr': '[1.,  0, 0]',                     \
 'MOD_rescal_pr': '[1.,  0.01, 0.693147181]'},       \
                                                     \
{'title': 'shear_any',                               \
 'BAS_shear_pr': '[0., -0.001, 0.001]',              \
 'MOD_transf_pr': '[1.,  0, 0]',                     \
 'MOD_shear_pr': '[1.,  -3.5, 3.5]'},                \
                                                     \
]

def write_experim(experm, filename, first4modorbas):
    writtens = {}
    for key in pr_keys:
        writtens[key] = False

    with open(filename, 'w') as outfile:
        # first, write title of experiment
        if 'title' in experm:
            outfile.write('title: '+experm['title']+'\n')
        # then write parameters (and check that they are formatted correctly)
        for key in experm:
            assert key == 'title' or key.startswith('BAS_') or key.startswith('MOD_'), key
            if key != 'title':
                assert key[4:] in pr_keys, key
                if key.startswith(first4modorbas):
                    outfile.write('t__'+key[4:]+' = '+experm[key]+'\n')
                    assert key[4:] in writtens, key
                    writtens[key[4:]] = True
        # finally, write the baseline values of any parameters not specified
        for key in pr_keys:
            if not writtens[key]:
                outfile.write('t__'+key+' = '+pr_key_to_baseline[key]+'\n')

exper_num = 0
for nangles in [18,]:
    for psize in [64,]:
        for my_exper in experims:
            if exper_num is not None: #in [1,2,5]:

                #os.system('rm -f train_tmp.hdf5')
                #os.system('rm -f train_tmp.pkl')
                #os.system('rm -f valid_tmp.hdf5')
                #os.system('rm -f valid_tmp.pkl')

                titlestr = 'score_dfft'+str(int(doublefftnorm))+'_hcf'+str(int(handcraftedfeats))+'_a'+str(int(nangles))+'_psiz'+str(int(psize))+'_'+my_exper['title']+'_expm'+str(exper_num)

                write_experim(my_exper, 'transform_parms_baseline.txt', 'BAS_')
                write_experim(my_exper, 'transform_parms_detectme.txt', 'MOD_')

                precompargs = titlestr+' ' + str(int(nangles)) + ' ' + str(int(psize)) \
                            +' '+str(int(doublefftnorm))+' '+str(int(handcraftedfeats))

                os.system('python KLT_LDA_features_precompute.py '+os.path.join(UCID_path, UCID_train)+' train_tmp_'+precompargs)
                os.system('python KLT_LDA_features_precompute.py '+os.path.join(UCID_path, UCID_val)  +' valid_tmp_'+precompargs)

                print("precomputed for \'"+titlestr+"\'")

                if handcraftedfeats:
                    os.system('python KLT_LDA_features_classify_after_PCA.py' \
                                +' train_tmp_'+titlestr \
                                +' valid_tmp_'+titlestr \
                                +' '+titlestr+'.txt' \
                                +' '+my_exper['title'])
                else:
                    #os.system('python '+os.path.join(classifierpath,'handwriting_convnet_new.py')+' 1')
                    print("did not train a classifier, since NOT using handcrafted feats")

                #quit()

            exper_num += 1

print("KLT_LDA_iterate_features: done!")

