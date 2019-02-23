import os,sys
thispath = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import cv2
from random import shuffle,randint
CUDA_RADON_TRANSF_PATH = '/path/to/cuda-radon-transform'
sys.path.append(CUDA_RADON_TRANSF_PATH)
from dataset_medifor_patches import new_transf_params, transf_splice_object, update_transf_parms
from utils import imshow_nottoobig
 
def has_image_extension(fname):
    f = fname.lower()
    if len(f) <= 4:
        return False
    return f[-4:] in ['.png', '.tif', '.jpg', '.bmp'] or f.endswith('.jpeg') or f.endswith('.tiff')
def files_in_folder(folder):
    return [os.path.join(folder,f) for f in os.listdir(folder) \
        if os.path.isfile(os.path.join(folder,f)) \
        and has_image_extension(f)]
 
# variables relating to the       "pristine"        image start with "p"
# variables relating to the "spliced object source" image start with "s"
# variables relating to the        "output"         image start with "o"
 
def buildsplice(pfile, sfile, transformdict):
    pim = cv2.imread(pfile, cv2.IMREAD_COLOR)
    assert pim is not None and pim.size > 1 and len(pim.shape) == 3 and pim.shape[2] == 3, str(pfile)
 
    sim = cv2.imread(sfile, cv2.IMREAD_UNCHANGED)
    assert sim is not None and sim.size > 1 and len(sim.shape) == 3 and sim.shape[2] == 4, str(sfile)
 
    return transf_splice_object(pim, sim, transformdict) #, verbose=True)
 
 
def build_splice_dataset(pdir, odir):
    assert os.path.exists(pdir), pdir
    if not os.path.exists(odir):
        os.makedirs(odir)
        assert os.path.exists(odir), odir
 
    sdir = os.path.join(thispath,'output_masked_objects')
 
    splic_srcs = files_in_folder(sdir)
    prist_srcs = files_in_folder(pdir)
    assert len(splic_srcs) > 1 and len(prist_srcs) > 1, str(len(splic_srcs))+', '+str(len(prist_srcs))
    #assert len(splic_srcs) > len(prist_srcs), str(len(splic_srcs))+', '+str(len(prist_srcs))
 
    shuffle(splic_srcs)
    shuffle(prist_srcs)
 
    iidx = 0
    splicidx = 0
    for jj in range(len(prist_srcs)):
        print prist_srcs[jj]
        for _ in range(0,6):
            doagain = True
            ii = np.random.randint(0,high=len(splic_srcs))
            while doagain:
                try:
                    res_rgb, res_mask = buildsplice(prist_srcs[jj], splic_srcs[ii], new_transf_params(1,None))
                    doagain = False
                except: # sometimes it will crash, if the splice object is larger than the background pristine image
                    pass
                splicidx += 1
     
            assert res_rgb.dtype == np.uint8
            assert res_mask.dtype == np.uint8
     
            cv2.imwrite(os.path.join(odir,str(iidx)+'_rgb.png'),   res_rgb)
            cv2.imwrite(os.path.join(odir,str(iidx)+'_mask.png'), 255-res_mask)
    
            print iidx 
            iidx += 1
 
 
if __name__ == "__main__":
    try:
        pdir = sys.argv[1]
        odir = sys.argv[2]
        assert os.path.exists(pdir)
    except:
        print("usage:  {folder-with-pristine}  {output-folder}")
        print(" Note: the \"output_masked_objects\" folder in this directory should contain the output of \"coco_get_seg_masks.py\"")
        quit()
    build_splice_dataset(pdir, odir)
