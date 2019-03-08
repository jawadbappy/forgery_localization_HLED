
## Localization of Image Forgeries

This project presents a framework to localize the image manipulation from an image. The network employs resampling features, Long-Short Term Memory (LSTM) cells, and encoder-decoder network in order to segment out manipulated regions from an image. 


### Data
We create a large dataset by splicing different objects obtained from MS-COCO dataset into the images of DRESDEN benchmark. Please check "synthetic_data" folder for more details.


### Model
Model can be found in "./model" folder. Please note that the given model is obtained by finetuning the base model with NIST data. The base model is trained on synthesized data.


### Resampling Features 
The codes for extracting resampling features can be found on "Radon" folder. Please change the input and output directory for your own dataset. Following is the command to extract the resampling features.

```
python extract_resamp_feat.py
```
In this code, the images are stored in hdf5 format. Please note that the package "pyfftw" are required to be installed before running the script. Please use the following command to install the package.  
```
sudo pip install pyfftw
```

### Train

First, the data needs to be prepared either hdf5 format or any other formats. The training code needs to be modified accordingly. In order to train the model, an image and a corresponding binary mask is required.   

```
python train.py
```
The model will be stored in the model path. 

### Test
We provide 8 sample images which will be found on test_data folder in order to demonstrate the output of the network. Please use the following command. 
```
python test.py
```
The code will automatically generate the binary mask and the heat map of prediction score.

### Sample Outputs
![Screenshot](output.png)
![Screenshot](output1.png)

### Citation
Please cite the following paper for reference. 
```
@article{bappy2019hybrid,
  title={Hybrid LSTM and Encoder-Decoder Architecture for Detection of Image Forgeries},
  author={Bappy, Jawadul H and Simons, Cody and Nataraj, Lakshmanan and Manjunath, BS and Roy-Chowdhury, Amit K},
  journal={IEEE Transactions on Image Processing},
  year={2019},
  publisher={IEEE}
}
```

