# Method 1: One Stage pass with 3 Yolov11 Models
## Each model trained on different conditions
1. Upload in kaggle notebook
- yolov11-one-stage-scs.ipynb
- yolov11-one-stage-nfn.ipynb
- yolov11-one-stage-ss.ipynb
- yolov11-one-stage-evaluation.ipynb
2. use the rsna 2024 lumbar spine degenerative classification competition input dataset and lsdc-utils datasets
3. use data fold splits from LSDC Gen Yolo Data SCS/NFN/SS notebooks from kaggle add input respectively and LSDC Fold split from kaggle add input for the train and valid dataset splits 
4. Then run the kaggle notebook
5. Save the weights respectively
6. take the weights for each model and Upload weights to yolov11-one-stage-evaluation.ipynb in kaggle
7. download ultralyics, import, LSDC Get All Images and LSDC Fold Split notebooks from kaggle add input
8. Run the one stage evaluation notebook to get submission.csv for the competition