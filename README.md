# Segmentation
## Preprocessing
* Download data from Kaggle (https://www.kaggle.com/c/data-science-bowl-2018/data) using the Kaggle API
* Unzip `stage1_train.zip` and extract to `Dataset/stage1_train`. We will be splitting this training set into train, validation and testing subsets in the training script `train.py`.
* cd to the repository directory
* Run `preprocess_dsb2018.py`

## Training
* Run `train.py --dataset dsb2018_96`
