# car_prediction
A deep learning model capable of predicting if two cars are of the same model

# Model + Training
1. Overfeat model is used as pretrained CNN
2. Logistic loss is used as training loss
3. Classification model in Section 4.1.1 is used as feature extractor and then a Joint Bayesian is applied to train a verification model on Part-II data
4. The performance of the model is tested on Part III of the data, which includes 1,145 car models

# Testing
1. Each image pair is classified by comparing the likelihood ratio produced by Joint Bayesian with a threshold
