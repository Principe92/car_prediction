# SVM

## Training + Validation test results
1. Trial 1: Initial
```
learning_rate = 0.1
epochs = 100
n_class = 2
rc = 0.01
SVM Training Accuracy : 58.872934 | Validation Accuracy: 59.024195
Notes: At epoch equals 50, the learning rate was reduced by a factor of 5
```

2. Trial 2: Remove learning rate adjustments
```
learning_rate = 0.1
epochs = 100
n_class = 2
rc = 0.01
SVM Training Accuracy : 65.721555 | Validation Accuracy: 65.441610
Notes: Looking at the graph, the loss increased after epoch 20
```


3. Trial 3: Remove learning rate adjustments
```
learning_rate = 0.1
epochs = 100
n_class = 2
rc = 0.001
SVM Training Accuracy : 65.721555 | Validation Accuracy: 65.441610
Notes: Looking at the graph, the loss increased after epoch 20
```