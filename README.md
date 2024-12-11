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


# Method 2: Two Stage pass using output from 3 Yolov11 Models into one ResNet50 model
1. Uploaded combined_output.csv file for all 194k+ images
2. Uploaded modified_file.csv for 61k+ images that are valid
3. Uploaded Stage2.ipynb which does the csv file modification and training of ResNet50 model
4. Uploaded cv_processing.ipynb which does the image cropping based on region proposals (checks for invalid boxes as well; may need improvements to filtering or thresholding)

## Steps
1. Use cv_processing.ipynb to crop the images using combined_output.csv as reference for bounding boxes
2. Use Stage2.ipynb to remove lines from combined_output.csv to get modified_file.csv based on cropped_images for images that are not saved in step 1
3. Run training phase of Stage2.ipynb to train model and get the weights.

# Streamlit App
## To test/inference, We have created a streamlit app that can be run
### Don't forget to download any modules needed by the streamlit
Modules:
- streamlit
- PIL
- numpy
- tensorflow
- ultralytics
- torch and torchvision
- matplotlib
- cv2

Steps to Run:
1. Download the `weights` folder and `Detect.py` and combine it into one folder
2. Run `Detect.py` in the terminal from the folder location by using
   
   ```Console
   streamlit run Detect.py
   ```
3. Once open in your browser, start uploading the image you want to test and wait for results.
