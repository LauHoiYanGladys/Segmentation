# Segmentation
## Preprocessing
* Download data from Kaggle (https://www.kaggle.com/c/data-science-bowl-2018/data) using the Kaggle API
* Unzip `stage1_train.zip` and extract to `Dataset/stage1_train`. We will be splitting this training set into train, validation and testing subsets in the training script `train.py`.
* cd to the repository directory
* Run `preprocess_dsb2018.py`

## Training
* Run `python train_new.py --dataset dsb2018_96 --arch UNet --epochs 20 --optimizer 'SGD' --lr 0.001 --scheduler ReduceLROnPlateau --patience 5 --factor 0.5 --batch_size 32`

## Validation
* Check the name of the new folder created in `model` after `train_new.py` was run above. The name is the `run_id`
* Run `pip install -U PyYAML`. This is needed for FullLoader in `yaml.load(f, Loader=yaml.FullLoader)` in `val.py`
* Run `python val.py --run_id (insert run_id)`

## Results
Result images can be found here: https://drive.google.com/drive/folders/11cL1fjtuW6iQce9gMpDM8QV78wQB92dh?usp=sharing

IOU metrics:
* Train IOU: 0.7245
* Validation IOU: 0.7512
* Test IOU: 0.7305
