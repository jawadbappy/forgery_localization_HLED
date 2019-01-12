# Using features
To extract the patch features:

- Extract patches using ```skimage.util.view_as_windows```

- Reshape patches to 4D array of shape [npatches, patchwidth, patchwidth, 3]

- The latest patch feature function is ```process_radon_input```
from ```KLT_LDA_features_precompute.py```.
For an example use see ```KLT_LDA_patch_feature_vector.py```.

- The script ```KLT_LDA_patch_feature_vector_keras.py``` is
similar to ```KLT_LDA_patch_feature_vector.py``` but uses a Keras-based
neural network as the feature classifier. Both use ```process_radon_input``` to
extract features, they differ in the classifier they apply to those features.

There are a lot of scripts in this repository, most are for visualization or
machine learning experiments.
The latest and most useful scripts were described above.

# Training and evaluating models

### Training a model

To train a model on the six classifiers,
simply run the script ``KLT_LDA_iterate_features.py`` and pass argument ``1``.
If you pass argument ``0`` (no handcrafted features), it will only extract and
resample the patches, and you will need to train an end-to-end model,
which is *not* in this repository.
Otherwise pass ``1`` to train all 6 classifiers automatically and save outputs
in the current directory.

To save space, delete any files starting with ``train_tmp_`` or ``valid_tmp_``,
which are the precomputed features on the patches;
unless you passed ``0`` (then you need them to train the end-to-end model).

You will need two dataset files in hdf5 format,
one with training patches and one with validation patches.
They will be loaded by ``KLT_LDA_features_precompute.py``, which loads
each as a large array, in one of two shapes:
4-dimensional [num-patches, width, height, channels], or
5-dimensional [num-images, patches-per-image, width, height, channels].

### Running a trained model

When you have a trained model, and the classifier files are moved into a folder,
run ``KLT_LDA_patch_feature_vector_keras_BATCH.py``. This is a parallelized
batch launcher for ``KLT_LDA_patch_feature_vector_keras.py`` which launches it
3 times in 3 processes on the same GPU.
Each process needs up to 20-23 GB memory; it works fine on a machine with 64 GB
of memory as long as you have at least 8 GB swap space (going over 64 GB is rare
  enough that the slower swap space doesn't become a performance problem).
Make sure you don't run more than 3 processes on the same GPU,
or it will be unstable and can crash at any time, often 20 to 60 minutes later.
On a machine with more than one GPU, you will also need to specify which GPU to
run on, with the code ``os.environ['CUDA_VISIBLE_DEVICES'] = '0'`` which is at
the top of the file ``KLT_LDA_patch_feature_vector_keras.py``, and in this case
is ensuring the code runs on GPU 0.
Its arguments ask for:
- The folder with classifier files output by
``KLT_LDA_iterate_features.py`` (but not ``train_tmp_*`` nor ``valid_tmp_*``).
- The expected fraction for the probability of resampling occurring.
I tried 0.15 to 0.3; try not to use a value too small because the
  outputs will become very dark as the average predicted value becomes small.
- When it asks for ``{image-filename(s)...}``, that means it is asking for
the specific file names, not a directory.
Simply use a wildcard, for example ``/data/nc2017/*.jpg``, to run many images.

# Build and installation

How to build, on Ubuntu 16.04:

   ```bash
   ./build_opencv_2.sh
   . export_opencv_2.sh
   ./ubuntu_requirements.sh
   ./build.sh
   ```

The C++ code requires CUDA, boost-python, libpython-dev, and OpenCV 2.4.x.
It does NOT work with OpenCV 3 due to it changing the cv::Mat backend.

If you use Anaconda Python, see "anaconda_requirements.txt" for packages to
install. You MUST install OpenCV and boost-python using Anaconda,
with "conda install", AND add prepend their paths (which will be
in the Anaconda directories) to your environment variables.

There are many required Python packages, some of which are:

numpy
scipy
matplotlib
skimage
tables

Some scripts require cv2 (python-opencv) and Tensorflow.
