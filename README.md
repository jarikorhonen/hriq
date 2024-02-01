# High Resolution Image Quality (HRIQ) database and model

## The database
The resolutions of the images used for currently available large scale natural image quality databases tend to be relatively low. To bridge this gap, we present High Resolution Image Quality (HRIQ) database with 1,120 images with natural distortions, such as sensor noise, motion blur, and focus blur. The images have been rated by 175 test persons (about 25 ratings per image) in a laboratory environment.

The database and the related metadata with Mean Opinion Scores (MOS) are available for download [here](https://drive.google.com/drive/folders/1TUaAD0pQMZjJInDpWd_apYXh3kDF6ydA). We have included three different resolution versions of each image in the dataset: 2880x2160, 1024x768, and 512x384; however, note that only the largest resolution was used in subjective testing. We will add a link to the paper with more details about the database here later.

If you use this database in your research, please include citation: ***H. Huang, Q. Wan, J. Korhonen: "High Resolution Image Quality Database," ICASSP'24, Seoul, South Korea, April 2024 (accepted for publication), preprint available [here](https://arxiv.org/abs/2401.16087)***. 

## The model

### Technical description
The model consists of three main parts: 1) pre-processing with formation of two streams for the original resolution and downsampled resolution as well as extraction of patches, 2) feature extraction using patches as input, and 3) spatial pooling and regression. The pre-processing stage is essentially similar to the corresponding stage in our earlier proposed BIQA models [RNN-BIQA](https://github.com/jarikorhonen/rnnbiqa) and [RNN-BIQAv2](https://github.com/jarikorhonen/rnnbiqav2). The assessed image is used as input to two streams, one in original resolution and one downsampled by factor of two in both horizontal and vertical dimensions. Then, the image in both streams is divided in fixed size patches of 224x224 pixels. The patches are sampled with equal intervals so that the whole spatial area of the input image is covered. If vertical or horizontal resolution is not divisible by 224, the patches will be partially overlapping.

The feature extractor is used to extract a feature vector from each input patch; in other words, the feature extractor converts a sequence of patches into a sequence of feature vectors. In RNN-BIQA, we used a modified ResNet50 network fine-tuned for BIQA as a feature extractor. However, related studies have shown that including semantic information in a BIQA model can improve image quality prediction accuracy. ViT has achieved impressive results in image classification, indicating that ViT features represent semantic image content more accurately than CNN features. On the other hand, some studies suggest that ViT is not optimal for extracting low level image features that are essential for image quality perception. Therefore, we have chosen to use a vanilla ViT in parallel with modified ResNet50 in our model design. The feature extractor forms the output feature vector by concatenating the feature vectors from the modified ResNet50 (2176 elements), and ViT (768 elements). In this architecture, the CNN feature vector represents the low level quality features, whereas the ViT feature vector represents the semantic features of the patch content.  

Finally, RNN is used for spatial pooling and regression. The proposed RNN model has similar design as in RNN-BIQAv2, with some differences. The features extracted from the low and high resolution patches form two sequences of feature vectors that are used as input to the two branches of the RNN model. Both branches include a fully connected layer with shared weights for prescaling, three GRU layers with individual weight for sequential processing, and finally a GRU head for regression after weighted averaging and concatenation of the feature sequences from the two branches.

### Usage
The usage of the model is similar to RNN-BIQAv2, except that in this case we use HRIQ dataset instead of third-party datasets. For fine-tuning the CNN model, you need to have LIVE Challenge image quality database installed from: http://live.ece.utexas.edu/research/ChallengeDB/.

For training and testing the model from scratch, you can use `masterScript.m`. It can be run from 
Matlab command line as:

```
>> masterScript_hriq(livec_path, koniq_path, spaq_path, cpugpu);
```

The following input is required:

`livec_path`: path to the LIVE Challenge dataset, including metadata files _allmos_release.mat_ and 
_allstddev_release.mat_. For example: _'c:\\livechallenge'_.

`hriq_path`: path to the HRIQ dataset, including metadata files _hriq_mos_file.csv_. For example: _'c:\\hriq'_.

`cpugpu`: whether to use CPU or GPU for training and testing the models, either _'cpu'_ or _'gpu'_.

The script implements the following functionality:

1) Makes patches out of LIVE Challenge dataset and makes probabilistic quality scores (file 
_LiveC_prob.mat_), `using processLiveChallenge.m` script.
3) Trains CNN feature extractor, using `trainCNNmodelV3.m` script.
4) Extracts feature vector sequences from HRIQ images, using the trained
feature extractor, off-the-shelf ViT model and `computeHRIQModelFeatures.m` script. *Note that for using the ViT model, you need to use Matlab version R2023b or later!* 
5) Trains and tests RNN model by using HRIQ with ten different splits to training and testing data. Uses `trainAndTestHRIQmodel.m` script for this purpose. Displays the results for SCC, PCC, and RMSE after each iteration, and finally the average results.
