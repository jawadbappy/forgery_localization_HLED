## Synthetic Data
For this project we synthesized approximately 170K manipulated images by splicing objects from the MS_COCO dataset onto known unmanipulated images.
Most, but not all due to size contraints, of these images can be found here: https://www.dropbox.com/sh/palus3sq4zvdky0/AACu3s7KA5Fhr_BJUeDOxnTLa?dl=0

## Datasets Description
The released dataset is split between three different zip files described below.

spliced_nist.zip - ~27K, Pristine images from the MFC18, spliced with a single object from the MS_COCO dataset 
copymove_nist.zip - ~27K, Pristine images from the MFC18, with a single object from MS_COCO spliced onto it multiple times
dresden_spliced_mscoco.zip - ~71K, Pristine images cropped from the Dresden Dataset, spliced with objects fromthe MS_COCO dataset

Each zip files contains files of the format [0-9]_mask.png and [0-9]_rgb.png. The rgb image is the spliced image and the mask file with the same number is the corresponding ground truth mask

## Creating More Data
Additional synthetic images can be created using the python files.

-splice_masks_copymove.py
 $python splice_masks_copymove {folder-with-pristine}  {output-folder}

-splice_masks_onto_pristine_images_2.py
 $python splice_masks_onto_pristine_images_2.py {folder-with-pristine}  {output-folder}

### Citation
Please cite the following paper if you use the data or the codes. 
```
@article{bappy2019hybrid,
  title={Hybrid LSTM and Encoder-Decoder Architecture for Detection of Image Forgeries},
  author={Bappy, Jawadul H and Simons, Cody and Nataraj, Lakshmanan and Manjunath, BS and Roy-Chowdhury, Amit K},
  journal={IEEE Transactions on Image Processing},
  year={2019},
  publisher={IEEE}
}
```
