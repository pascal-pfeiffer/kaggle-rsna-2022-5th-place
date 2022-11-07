# Kaggle RSNA 2022 - 5th place solution

Thanks to kaggle and the sponsors for organizing this interesting and leak-free competition which had a lot of different angles to explore. Coming directly from the DFL and joining quite late here, we were explicitly looking for the sprint aspect and to challenge ourselves to derive a good solution in only 11 days. Hence our team name: Speedrun (Philipp, Pascal, Christof)
Luckily, we could build on ideas shared from previous 3D-RSNA competitions as well as competitions we did before. 


Our solution is an ensemble of two quite similar approaches which follow a 3-stage paradigm:

- Train a 2D classification/ segmentation model for vertebrate using the provided segmentation labels
- Use the resulting model to predict the vertebrate class visible in each all dicom images and multiply the result with the overall fracture label to get 2D image level labels if a certain vertebrae is fractured or not.
- Collect all image level labels per study and train an aggregation model to predict the given study level labels. 

1. Download and extract the competition data to `./input/rsna-2022-cervical-spine-fracture-detection/`
2. Download and extract metadata from `https://www.kaggle.com/datasets/samuelcortinhas/rsna-2022-spine-fracture-detection-metadata` to `./input/rsna-2022-spine-fracture-detection-metadata/`
3. Run `sh run_stage1.sh` for preprocessing and training the 2D classification & segmentation 1st stage models
4. Run `sh run_stage2.sh` for preprocessing and training the 2nd stage models
4. Run `sh run_stage3.sh` for training the 3rd stage models
