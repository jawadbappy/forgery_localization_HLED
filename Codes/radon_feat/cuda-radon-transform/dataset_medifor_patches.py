"""
Author: Jason Bunk

Helpers for building a dataset of transformed patches.
"""
import os,sys
thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(thispath,'build'))
#radonpath = os.path.join(thispath,'../MEDIFOR/cuda-radon-transform')
#assert os.path.exists(radonpath), radonpath
#sys.path.append(radonpath)
from radon_transform_features import radon_transform_features
import time
import numpy as np
import numpy.random as npr
import cv2
#import vizutils
import tables
from utils import *
from splice_utils import *

def build_transf_mat(rth, shA, shB, scX, scY, cX, cY, trX, trY):
    rth *= (np.pi / 180.) # to radians
    rcos = np.cos(rth)
    rsin = np.sin(rth)
    shA /= cX # normalize shearing
    shB /= cY
    return np.array([[scX*rcos + shB*rsin, shA*rcos + scY*rsin, -(scX*rcos + shB*rsin)*cX - (shA*rcos + scY*rsin)*cY + cX + trX], \
                     [shB*rcos - scX*rsin, scY*rcos - shA*rsin, -(shB*rcos - scX*rsin)*cX - (scY*rcos - shA*rsin)*cY + cY + trY]])

t__transf_pr  = [0.4, None, None]
t__splicecolor_pr  = [0.30, None, None]
t__spliceedgesm_pr  = [0.30, None, None]
t__jpgpre_pr  = [0.95, 65, 100]
t__jpgpost_pr = [0.95, 65, 100]
t__angles_pr  = [0.5, -90., 90.]
t__shear_pr   = [0.2, -3.5, 3.5]
t__rescal_pr  = [0.6, -0.693147181, 0.693147181]

def update_transf_parms(filename):
    global t__transf_pr
    global t__splicecolor_pr
    global t__spliceedgesm_pr
    global t__jpgpre_pr
    global t__jpgpost_pr
    global t__angles_pr
    global t__shear_pr
    global t__rescal_pr
    parmschanged = {'t__transf_pr':  None, \
                    't__splicecolor_pr':  None, \
                    't__spliceedgesm_pr':  None, \
                    't__jpgpre_pr':  None, \
                    't__jpgpost_pr': None, \
                    't__angles_pr':  None, \
                    't__shear_pr':   None, \
                    't__rescal_pr':  None}
    assert os.path.isfile(filename), filename
    with open(filename, 'r') as infile:
        for line in infile:
            line = line.strip()
            if '[' in line and ']' in line and '=' in line and ',' in line:
                assert line.count('[') == 1 and line.count(']') == 1 and line.count('=') == 1, line
                assert line.find('=') < line.find('[') and line.find('[') < line.find(']') and line.find('[') < line.find(','), line
                spl = line.split('[')[-1].split(']')[0].replace(' ','').split(',')
                assert len(spl) == 3, str(spl)
                varname = line.split('=')[0].strip()
                newpr = [float(ss) for ss in spl]

                assert varname in parmschanged, line
                assert parmschanged[varname] is None, line
                parmschanged[varname] = newpr

                if varname == 't__transf_pr':
                    t__transf_pr = newpr
                elif varname == 't__splicecolor_pr':
                    t__splicecolor_pr = newpr
                elif varname == 't__spliceedgesm_pr':
                    t__spliceedgesm_pr = newpr
                elif varname == 't__jpgpre_pr':
                    t__jpgpre_pr = newpr
                elif varname == 't__jpgpost_pr':
                    t__jpgpost_pr = newpr
                elif varname == 't__angles_pr':
                    t__angles_pr = newpr
                elif varname == 't__shear_pr':
                    t__shear_pr = newpr
                elif varname == 't__rescal_pr':
                    t__rescal_pr = newpr
                else:
                    assert 0, line

    nchanged = 0
    for key in parmschanged:
        if parmschanged[key] is not None:
            nchanged += 1
    print("read "+str(nchanged)+" new params from \'"+str(filename)+'\'')
    for key in parmschanged:
        print("        "+key+" = "+str(parmschanged[key]))

def new_transf_params(bs, num2, checktransf=False):
    global t__transf_pr
    global t__splicecolor_pr
    global t__spliceedgesm_pr
    global t__jpgpre_pr
    global t__jpgpost_pr
    global t__angles_pr
    global t__shear_pr
    global t__rescal_pr

    bs = int(bs)
    if num2 is None:
        shape = (bs,)
    else:
        shape = (bs,int(num2))
    pd = {}
    zeros = np.zeros(shape=shape,dtype=np.int32)
    oness =  np.ones(shape=shape,dtype=np.int32)

    pd['jpgpre'] = npr.randint(int(t__jpgpre_pr[1]), int(t__jpgpre_pr[2]), size=shape) * npr.binomial(1, float(t__jpgpre_pr[0]), size=shape)

    pd['tran_do'] = npr.binomial(1, t__transf_pr[0], size=shape)
    pd['splicecolor'] = npr.binomial(1, t__splicecolor_pr[0], size=shape)
    pd['spliceedgesmooth'] = npr.binomial(1, t__spliceedgesm_pr[0], size=shape)
    pd['tran_angles']  = npr.uniform(t__angles_pr[1], t__angles_pr[2], size=shape) * npr.binomial(1, t__angles_pr[0], size=shape)
    pd['tran_shearAs'] = npr.uniform(t__shear_pr[1],  t__shear_pr[2], size=shape) * npr.binomial(1, t__shear_pr[0], size=shape)
    pd['tran_shearBs'] = npr.uniform(t__shear_pr[1],  t__shear_pr[2], size=shape) * npr.binomial(1, t__shear_pr[0], size=shape)
    pd['tran_rescalX'] = npr.uniform(t__rescal_pr[1], t__rescal_pr[2], size=shape) * npr.binomial(1, t__rescal_pr[0], size=shape)
    pd['tran_rescalY'] = npr.uniform(t__rescal_pr[1], t__rescal_pr[2], size=shape) * npr.binomial(1, t__rescal_pr[0], size=shape)
    pd['tran_tranX']   = npr.uniform(-4.5, 4.5, size=shape)
    pd['tran_tranY']   = npr.uniform(-4.5, 4.5, size=shape)
    pd['tran_interpt'] = npr.binomial(1, 0.333, size=shape)
    # ensure we actually are doing a transformation, if tran_do
    for kk in range(np.prod(shape)):
        idx = np.unravel_index(kk,shape)
        if checktransf and pd['tran_do'][idx]:
            theseprods = lambda pd,ii: np.amax(np.fabs(np.array([ \
                                        pd['tran_angles'][ii], pd['tran_shearAs'][ii], pd['tran_shearBs'][ii] \
                                                             , pd['tran_rescalX'][ii], pd['tran_rescalY'][ii] \
                                                                            ])))
            while theseprods(pd,idx) < 1e-6:
                pd['tran_angles'][idx]  = npr.uniform(t__angles_pr[1], t__angles_pr[2]) * npr.binomial(1, t__angles_pr[0])
                pd['tran_shearAs'][idx] = npr.uniform(t__shear_pr[1],  t__shear_pr[2]) * npr.binomial(1, t__shear_pr[0])
                pd['tran_shearBs'][idx] = npr.uniform(t__shear_pr[1],  t__shear_pr[2]) * npr.binomial(1, t__shear_pr[0])
                pd['tran_rescalX'][idx] = npr.uniform(t__rescal_pr[1], t__rescal_pr[2]) * npr.binomial(1, t__rescal_pr[0])
                pd['tran_rescalY'][idx] = npr.uniform(t__rescal_pr[1], t__rescal_pr[2]) * npr.binomial(1, t__rescal_pr[0])

    pd['translX'] = npr.randint(-4, 5, size=shape)
    pd['translY'] = npr.randint(-4, 5, size=shape)

    pd['jpgpost'] = zeros
    for kk in range(np.prod(shape)):
        idx = np.unravel_index(kk,shape)
        pd['jpgpost'][idx] = npr.randint(int(t__jpgpost_pr[1]), int(t__jpgpost_pr[2]))
        if pd['tran_do'][idx]:
            pd['jpgpost'][idx] *= npr.binomial(1, float(t__jpgpost_pr[0]))
        else:
            pd['jpgpost'][idx] *= npr.binomial(1, float(t__jpgpost_pr[0]) * 0.15)

    return pd


pdeltas = {
'jpgpre' : 10.,
'tran_do': 0.5,
'splicecolor': 0.5,
'tran_angles' : 30.,
'tran_shearAs': 1.4,
'tran_shearBs': 1.4,
'tran_rescalX': 0.21,
'tran_rescalY': 0.21,
'tran_tranX': 1.5,
'tran_tranY': 1.5,
'tran_interpt': 0.5,
'translX': 1.5,
'translY': 1.5,
'jpgpost': 10.
}
def fix_transform_compatibility_wrt_labels(labels, pd, jpgonly=False):
    for ii in range(labels.size):
        if not bool(labels[ii,0]):
            # label is 0, they are the same: transform second patch w/ same parameters
            for key in pd:
                pd[key][ii,1] = pd[key][ii,0]
        else:
            # label is 1, they are different: use different transformation for second patch
            while True:
                if bool(pd['tran_do'][ii,0]) == bool(pd['tran_do'][ii,1]):
                    aresame = True
                    if bool(pd['tran_do'][ii,0]):
                        for key in pd:
                            aresame = aresame and np.fabs(float(pd[key][ii,1]) - float(pd[key][ii,0])) < pdeltas[key]
                    else:
                        for key in pd: # not doing transformation, so don't compare transformation parameters
                            if not key.startswith('tran_'):
                                aresame = aresame and np.fabs(float(pd[key][ii,1]) - float(pd[key][ii,0])) < pdeltas[key]
                    if aresame:
                        newparms = new_transf_params(1,1,jpgonly=jpgonly)
                        for key in pd:
                            pd[key][ii,1] = newparms[key][0,0]
                    else:
                        break
                else:
                    break
    return pd


def transf_one_patch(patch, pd, patch_size, verbose=False):
    if verbose:
        describe("patch", patch)
        cv2.imshow("patch-RAW",patch[:,:,::-1])

    assert len(patch.shape) == 3 and patch.shape[2] == 3

    if npr.binomial(1,0.5):
        patch = patch[::-1, :, :]
    # randint upper end is exclusive so randint(0,4) ranges from [0,3] inclusive
    rotangle = npr.randint(0,4)
    if rotangle > 0:
        patch = np.rot90(patch, k=rotangle)

    assert len(patch.shape) == 3 and patch.shape[2] == 3

    if int(pd['jpgpre']) > 1:
        patch = cv2.imdecode( \
            cv2.imencode('.jpg',patch,params=[cv2.IMWRITE_JPEG_QUALITY,int(pd['jpgpre'])])[1], \
            cv2.IMREAD_COLOR)

    if verbose:
        cv2.imshow("patch-BEF",patch[:,:,::-1])

    if bool(pd['tran_do']):
        tmat = build_transf_mat(pd['tran_angles'], pd['tran_shearAs'], pd['tran_shearBs'], \
                            np.exp(pd['tran_rescalX']), np.exp(pd['tran_rescalY']), \
                            cX=0.5*float(patch.shape[1]), cY=0.5*float(patch.shape[0]), \
                            trX=float(pd['tran_tranX']), trY=float(pd['tran_tranY']))
        affflag = cv2.INTER_LINEAR
        if bool(pd['tran_interpt']):
            affflag = cv2.INTER_CUBIC
        patch = cv2.warpAffine(patch, tmat, dsize=(patch.shape[1],patch.shape[0]), flags=affflag, borderMode=cv2.BORDER_REFLECT_101)

    patch_size = int(patch_size)
    PSTR = (192-patch_size)//2
    PEND = PSTR + patch_size

    patch = patch[(PSTR+pd['translX']):(PEND+pd['translX']), \
                  (PSTR+pd['translY']):(PEND+pd['translY']), :]

    if int(pd['jpgpost']) > 1:
        patch = cv2.imdecode( \
            cv2.imencode('.jpg',patch,params=[cv2.IMWRITE_JPEG_QUALITY,int(pd['jpgpost'])])[1], \
            cv2.IMREAD_COLOR)

    if verbose:
        cv2.imshow("patch-AFT",patch[:,:,::-1])
        nangles = 180
        print("VERBOSE: COMPUTING RADON TRANSFORM FEATURES FOR "+str(nangles)+" ANGLES")
        pfeats = radon_transform_features(np.expand_dims(patch,0), numAngles=nangles, to3channel=True, multiprocess=True)
        cv2.imshow("pfeats",uint8norm(pfeats[0,:,:,-1]))
        cv2.waitKey(0)
    return patch


def transf_splice_object(img_background, img_object, pd, verbose=False):
    if verbose:
        describe("img_background", img_background)
        describe("img_object", img_object)
        cv2.imshow("img_object-RAW",img_object)

    assert len(img_object.shape) == 3 and img_object.shape[2] == 4, str(img_object.shape)

    if bool(pd['splicecolor']):
        im___a = img_object[:,:,3]
        im_rgb = splice_color_transfer(img_background, img_object[:,:,:3])
        img_object = np.concatenate((im_rgb, np.expand_dims(im___a,-1)), axis=2)

    assert len(img_object.shape) == 3 and img_object.shape[2] == 4, str(img_object.shape)

    if npr.binomial(1,0.5):
        img_object = img_object[::-1, :, :]
    # randint upper end is exclusive so randint(0,4) ranges from [0,3] inclusive
    rotangle = npr.randint(0,4)
    if rotangle > 0:
        img_object = np.rot90(img_object, k=rotangle)

    assert len(img_object.shape) == 3 and img_object.shape[2] == 4, str(img_object.shape)

    if int(pd['jpgpre']) > 1:
        im___a = img_object[:,:, 3]
        im_rgb = cv2.imdecode( \
            cv2.imencode('.jpg',img_object,params=[cv2.IMWRITE_JPEG_QUALITY,int(pd['jpgpre'])])[1], \
            cv2.IMREAD_COLOR)
        img_object = np.concatenate((im_rgb, np.expand_dims(im___a,-1)), axis=2)

    if verbose:
        cv2.imshow("img_object-BEF",img_object)
        print("pd['tran_do'] == "+str(pd['tran_do']))

    assert len(img_object.shape) == 3 and img_object.shape[2] == 4, str(img_object.shape)

    if bool(pd['tran_do']):
        shapX = 0.5*float(img_object.shape[1])
        shapY = 0.5*float(img_object.shape[0])
        tmat = build_transf_mat(pd['tran_angles'], pd['tran_shearAs'], pd['tran_shearBs'], \
                            np.exp(pd['tran_rescalX']), np.exp(pd['tran_rescalY']), \
                            cX=shapX, cY=shapY, \
                            trX=shapX+float(pd['tran_tranX']), trY=shapY+float(pd['tran_tranY']))
        affflag = cv2.INTER_LINEAR
        if bool(pd['tran_interpt']):
            affflag = cv2.INTER_CUBIC
        if img_object.shape[2] == 3:
            assert 0
            #img_object = cv2.warpAffine(img_object, tmat, dsize=(img_object.shape[1],img_object.shape[0]), flags=affflag, borderMode=cv2.BORDER_REFLECT_101)
        else:
            im_rgb = cv2.warpAffine(img_object[:,:,:3], tmat, dsize=(2*img_object.shape[1],2*img_object.shape[0]), flags=affflag, borderMode=cv2.BORDER_REFLECT_101)
            im___a = cv2.warpAffine(img_object[:,:, 3], tmat, dsize=(2*img_object.shape[1],2*img_object.shape[0]), flags=affflag, borderMode=cv2.BORDER_CONSTANT)
            img_object = np.concatenate((im_rgb, np.expand_dims(im___a,-1)), axis=2)

    assert len(img_object.shape) == 3 and img_object.shape[2] == 4, str(img_object.shape)

    if bool(pd['spliceedgesmooth']):
        im_rgb = img_object[:,:,:3]
        im___a = img_object[:,:, 3]
        im___a = cv2.GaussianBlur(im___a, (0,0), 1.0)
        img_object = np.concatenate((im_rgb, np.expand_dims(im___a,-1)), axis=2)

    assert len(img_object.shape) == 3 and img_object.shape[2] == 4, str(img_object.shape)

    img_object = crop_to_bounding_box_from_alpha(img_object)
    res_rgb, res_mask = splice_crop_into_image(img_background, img_object)

    if verbose:
        cv2.imshow("crop",img_object)
        cv2.imshow("crop-alpha",img_object[:,:,3])
        imshow_nottoobig("res_rgb",res_rgb)
        imshow_nottoobig("res_mask",res_mask)
        describe("res_rgb", res_rgb)
        describe("res_mask", res_mask)

    if int(pd['jpgpost']) > 1:
        res_rgb = cv2.imdecode( \
            cv2.imencode('.jpg',res_rgb,params=[cv2.IMWRITE_JPEG_QUALITY,int(pd['jpgpost'])])[1], \
            cv2.IMREAD_COLOR)

    return res_rgb, res_mask


def load_hdf5_dataset(fname):
    fname = str(fname)
    assert os.path.exists(fname), fname
    h5file = tables.open_file(fname,mode='r')
    dataset = h5file.root.dataset[:]
    h5file.close()
    print("unpickled dataset of shape "+str(dataset.shape))
    return dataset
