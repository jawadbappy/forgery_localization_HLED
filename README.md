# Image Forgery Localization
## Repo for Multi-pose Image Generation
The work presents a GAN based deep generative model in order to transfer a person's current pose to a new pose. 

### Data
The code utilizes DeepFashion dataeset. Please use the following link to download the DeepFashion images. 
```
http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
```
We also demonstrated rigorous experimentation on Market-1501 dataset. The test data (DeepFashion and Market-1501) have also been 
uploaded to the data server (10.252.194.1). Please check "multipose_data" folder in 10.252.194.1 server. The performance of model
has been evaluated in terms of Structural Similarity (SSIM) and Inception Score (IS). 

### Pose Estimation
In this work, pose keypoints have been estimated using the following method. 
```
https://github.com/last-one/Pytorch_Realtime_Multi-Person_Pose_Estimation
```

### Model
Model can be found in "multipose_data" folder in 10.252.194.1 server.

### Train
First, the config file needs to be modified to train the network.  

```
python train_multipose_generator.py --config /config file/ --output_path /out path/
```
The model will be stored in the output path. 

### Test
To test the image generation results, please use the following format. 
```
python pose_generation_test.py --config ./configs/config file --image_path 
/image path/  --out_path /results/ --model_path /model path/

```

### Score Generation
Following command is for the computation of SSIM and IS score. 

```
python score.py /model path/ /results/ dataset
```
Plese use 'df' or 'mk' to denote the DeepFashion or Market dataset. 
After running the above command, a file "ssim_is_score.txt" will be generated that includes the SSIM and IS score.
The results are provided in "ssim_is_score_df.txt" and "ssim_is_score_market.txt" for the DeepFashion or the Market 
datasets respectively. 


