#!/usr/bin/env python
"""
Author: Jason Bunk

Wrapper around KLT_LDA_patch_feature_vector_keras.py
which launches it multiple times in parallel to run on a batch of images.

The interface is identical to the above file;
the first few lines of the __main__ function should be identical.
"""
import os,sys
thispath = os.path.dirname(os.path.abspath(__file__))

import ntpath
from glob import glob
import multiprocessing
import subprocess

def filename_without_extension(fname):
    assert '.' in fname
    return fname[:fname.rfind('.')]

def isokayimgname(fname):
    basen = filename_without_extension(fname)
    return not basen.endswith('_baseunmanip')

def outnotexistyet(out_folder, fname):
    basename = os.path.join(out_folder, ntpath.basename(fname[:-4]))
    exst = os.path.isfile(basename+'_clf_0-3.png') \
       and os.path.isfile(basename+'_clf_4-6.png') \
       and os.path.isfile(basename+'_numpy.npy')
    return not exst

def chunkify(lst,n):
    return [ lst[i::n] for i in xrange(n if n < len(lst) else len(lst)) ]

if __name__ == '__main__':
    try:
        classiffolder = sys.argv[1]
        fraction_manip = float(sys.argv[2])
        out_folder = sys.argv[3]
        assert len(sys.argv) > 4
        assert fraction_manip > 0. and fraction_manip < 1.
    except:
        print("usage:  {folder-with-classifiers}  {fraction-manip}  {output-folder}  {image-filename(s)...}")
        quit()

    assert os.path.exists(classiffolder), classiffolder
    if not os.path.exists(out_folder):
        os.system('mkdir \''+out_folder+'\'')

    if not out_folder.endswith('/'):
        out_folder += '/'
    CLASSIF_FILEEXTN = '_KerasResult.h5'
    CLASSIF_MEANEXTN = '_KerasResult2.h5'

    classifs = [ff for ff in os.listdir(classiffolder) if ff.endswith(CLASSIF_FILEEXTN)]
    infileRAW = [fname for argv in sys.argv[4:] for fname in glob(argv)]
    infiles = [fname for fname in infileRAW if isokayimgname(fname) and outnotexistyet(out_folder,fname)]

    print(" ")
    print("num experiments: "+str(len(classifs)))
    print("num images given: "+str(len(infiles)))
    print(" ")

    NPROCESSES = 3
    infilechunked = chunkify(infiles, NPROCESSES)
    assert len(infilechunked) == NPROCESSES or len(infiles) < NPROCESSES
    assert sum([len(chnk) for chnk in infilechunked]) == len(infiles)

    def mapfun(lst):
        return subprocess.check_output(['python', 'KLT_LDA_patch_feature_vector_keras.py', sys.argv[1], sys.argv[2], sys.argv[3]]+lst)

    pool = multiprocessing.Pool(NPROCESSES)
    RETMAPPED = pool.map(mapfun, infilechunked)
    pool.close()
    pool.join()
    tryname = 'KLT_LDA_patch_feature_vector_keras_OUTPUT_'
    codenum = 0
    while os.path.isfile(tryname+str(codenum)+'.txt'):
        codenum += 1
    with open(tryname+str(codenum)+'.txt', 'w') as outfile:
        for mystr in RETMAPPED:
            outfile.write(mystr+'\n=============================\n')

