#Jpeg_artifact_removal

Included files are as follows:. 

1. preprocess_data.py -->preprocessed the data,creates augmentated data,normalizes and stores them in a folder

2. train_data.py --> Takes images from training_data.py,stores them in memory,builds a keras(tensorflow backend) model and trains them

3.  evaluate_model.py --> takes the validation data , predicts the high resolution images and calculates the psnr scores.
Just run this script to see the model predictions

4. model.h5 --> The actual model which was trained.Due to limitations of the machine, it couldnot be trained for long and so 
the psnr is not very good.The average psnr is coming to be 29.8(w.r.t. the high res image) .

