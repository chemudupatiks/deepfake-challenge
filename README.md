#  Deepfake detection challenge on Kaggle 

https://www.kaggle.com/c/deepfake-detection-challenge

This project uses the differences in consecutive frames to capture the synthetic transtitions added during the creation of deepfakes as a feature to train the model.
A simple CNN architecture of 2 Convolutional Layers and 2 Linear Layers gives a fairly good result of 75% accuracy for a very small subset of data (400 videos).

## Things to do
1. Train and test the model on more data
2. Optimize hyperparameters to improve the model
3. Try a combination of CNN + RNN 

### Notes
tf_model.py still needs work to be done. Currently not running due to issues with CUDA, I couldn't fix. 
