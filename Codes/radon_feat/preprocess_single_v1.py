
#used for file input and output
import os #used to read file folders
from skimage import io #read images
import tables as tb

#used to process images
from skimage.util import view_as_windows #for extracting patches
from skimage.transform import resize
from skimage.color import gray2rgb
from hilbert import hilbertCurve #used to generate n-order hilbert curves
import sys #used to add folder to path
sys.path.insert(0, '/home/mbappy/medifor/cuda-radon-transform/') #location of radon tfom
from radon_transform_features import radon_transform_features

#utility
import numpy as np
import argparse

#extract radon features from a set of patches at 10 angles
def extractRadonFeat(image_patches):
    radonFeatures = radon_transform_features(image_patches, numAngles=10,doublefftnorm=True)
    radonFeatures = np.squeeze(radonFeatures)
    return radonFeatures[:,0:10,:] #last element along axis 2 is not used for training

def openImage(file_path):
    img = io.imread(file_path)
    if len(img.shape) == 1: # Honestly not sure why this happens, but every so often shape = (2,)
        img = img[0]        # img[0] usually contains rgb img, best guess img[1] is alpha channel
    if len(img.shape) == 2: #grayscale image
        img = gray2rgb(img)
    return img

def main():
    parser = argparse.ArgumentParser(description='Preprocessing images for darpa challenge')
    parser.add_argument('input_dir', metavar='Input Directory', type=str,
                        help='Directory containing input images')
    parser.add_argument('output_file', metavar='Output File', type=str,
                        help='Full output file path')
    parser.add_argument('mode',choices=['training','eval'], type=str,
                        help='Mode that the data will be used for')
    parser.add_argument('--frac', type=float,default=0.2,
                        required=False,help='Percent of images to use for testing')
    parser.add_argument('--name', type=bool,default=False,
                        required=False,help='If true, image file names are stored as well')
    args = parser.parse_args()

    if args.mode == 'training':
        folders = sorted(os.listdir(args.input_dir))
        num_images = 0
        for i in xrange(len(folders)):        
           num_images += len(os.listdir(args.input_dir + '/' + folders[i]))

        hdf5 = tb.open_file(args.output_file,'w',\
                            'Medifor hilbert finetuning, 128x128 patch size')
        #Create EArrays that will hold the features and labels
        test_feat = hdf5.create_earray(hdf5.root, 'validation_features',
                                       tb.Float32Atom(),
                                       shape=(0, 64, 10, 92), # 10 is # of angles, 92 # of values for 128x128 patch
                                       expectedrows=num_images*args.frac) #64 patches per image
        test_label = hdf5.create_earray(hdf5.root, 'validation_labels',
                                        tb.Float32Atom(),
                                        shape=(0, len(folders)), # one hot encoding with two classes
                                        expectedrows=num_images*args.frac)
        train_feat = hdf5.create_earray(hdf5.root, 'training_features',
                                        tb.Float32Atom(),
                                        shape=(0, 64, 10, 92), # 10 is # of angles, 92 # of values for 128x128 patch
                                        expectedrows=num_images*(1-args.frac)) #64 patches per image
        train_label = hdf5.create_earray(hdf5.root, 'training_labels',
                                         tb.Float32Atom(),
                                         shape=(0, len(folders)), # one hot encoding with two classes
                                         expectedrows=num_images*(1-args.frac))
        if args.name:
            test_names = hdf5.create_earray(hdf5.root, 'validation_names',
                                            tb.StringAtom(32),
                                            shape(0,1),
                                            expectedrows=num_images*args.frac)
            train_names = hdf5.create_earray(hdf5.root, 'training_names',
                                             tb.StringAtom(32),
                                             shape(0,1),
                                             expectedrows=num_images*(1-args.frac))
                                            
        seq = np.linspace(0,63,64).astype(int)
        order3 = hilbertCurve(3)
        order3 = np.reshape(order3,(64))
        ind = np.lexsort((seq,order3)) 
    
        folders = sorted(os.listdir(args.input_dir))
        for i in xrange(len(folders)):
            images = sorted(os.listdir(args.input_dir + '/' + folders[i]))
            for j in xrange(len(images)):
                print './' + args.input_dir + '/' + folders[i] + '/' + images[j]
                rgb = openImage(args.input_dir + '/' + folders[i] + '/' + images[j])
                rgb = resize(rgb, [1024, 1024, 3])

                rgb_patches = view_as_windows(rgb,(128,128,3),128)
                rgb_patches = np.squeeze(rgb_patches)
                rgb_patches = np.reshape(rgb_patches,(64,128,128,3))
                radon_features = extractRadonFeat(rgb_patches)
                radon_features = radon_features[ind]
                radon_features = np.reshape(radon_features,(1,64,10,92))
        
                patch_labels = np.zeros((1,len(folders)))
                patch_labels[0,i] = 1
        
                if j < len(images)*args.frac:
                    test_feat.append(radon_features)
                    test_label.append(patch_labels)
                    if args.name:
                        test_names.append(np.reshape([images[j][0:32]],(1,1)))
                else:
                    train_feat.append(radon_features)
                    train_label.append(patch_labels)
                    if args.name:
                        train_names.append(np.reshape([images[j][0:32]],(1,1)))
        hdf5.close()
    else:
        num_images = len(os.listdir(args.input_dir))
        hdf5 = tb.open_file(args.output_file,'w',\
                            'Medifor hilbert finetuning, 128x128 patch size')
        #Create EArrays that will hold the features and labels
        feature = hdf5.create_earray(hdf5.root, 'features',
                                          tb.Float32Atom(),
                                          shape=(0, 64, 10, 92), # 10 is # of angles, 92 # of values for 128x128 patch
                                          expectedrows=num_images) #64 patches per image
        if args.name:
            names = hdf5.create_earray(hdf5.root, 'names',
                                       tb.StringAtom(32),
                                       shape=(0,1),
                                       expectedrows=num_images)
        seq = np.linspace(0,63,64).astype(int)
        order3 = hilbertCurve(3)
        order3 = np.reshape(order3,(64))
        ind = np.lexsort((seq,order3))
#	print ind
        images = sorted(os.listdir(args.input_dir))
        for i in xrange(len(images)):
            print '/' + args.input_dir + '/' + images[i]
            rgb = openImage(args.input_dir + '/' + images[i])
            rgb = resize(rgb, [1024, 1024, 3])

            rgb_patches = view_as_windows(rgb,(128,128,3),128)
            rgb_patches = np.squeeze(rgb_patches)
            rgb_patches = np.reshape(rgb_patches,(64,128,128,3))
            radon_features = extractRadonFeat(rgb_patches)
            radon_features = radon_features[ind]
            radon_features = np.reshape(radon_features,(1,64,10,92))

            feature.append(radon_features)
            if args.name:
                names.append(np.reshape([images[i][0:32]],(1,1)))

        hdf5.close()


if __name__ == '__main__':
    main()
