#!/usr/bin/env python
import os,sys

exe = './KLT_LDA_patch_feature_vector_keras_BATCH.py'

def is_rgbpng(fname):
    return fname.endswith('_rgb.png')
def is_imgname(fname):
    ff = fname.lower()
    return ff[-4:] in ['.png', '.jpg', '.tif', '.bmp'] or ff.endswith('.jpeg') or ff.endswith('.tiff')

def run_on_folder(folder, checkfun=is_rgbpng):
    assert os.path.exists(folder), folder
    thefiles = ' '.join([os.path.join(folder,ff) for ff in os.listdir(folder) if checkfun(ff)])
    os.system(exe+' apr18__a18_psiz64/ 0.2 '+folder+'_resamp '+thefiles)


for ii in range(5):
    run_on_folder('/data/jbunk/mscoco/new_apr_2017/tmp/'+str(ii).zfill(4)+'_manip_splice')

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("done with manip_splice, next will do pristine")
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

for ii in range(5):
    run_on_folder('/data/jbunk/mscoco/new_apr_2017/tmp/'+str(ii).zfill(4)+'_pristine', checkfun=is_imgname)

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("done")
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
