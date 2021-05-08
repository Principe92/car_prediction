# SVM

## All training data results

1. Trial 1
```
learning_rate = 0.001
epochs = 50
n_class = 2
rc = 0.1
accuracy: 52
```

2. Trial 2: Reduce the learning rate
```
learning_rate = 0.01
epochs = 50
n_class = 2
rc = 0.1
accuracy: 79.200359
```

## Training + Validation test results
1. Trial 1: Initial
```
learning_rate = 0.01
epochs = 50
n_class = 2
rc = 0.1
Training Accuracy : 75.280899 | Validation Accuracy: 73.094170
```

2. Trial 2: Increase the epoch to 100
```
learning_rate = 0.01
epochs = 100
n_class = 2
rc = 0.1
Training Accuracy : 81.797753 | Validation Accuracy: 77.802691
```

3. Trial 3: Increase the learning rate
```
learning_rate = 0.1
epochs = 100
n_class = 2
rc = 0.1
Training Accuracy : 86.292135 | Validation Accuracy: 78.699552
```

4. Trial 4: Reduce the regularization constant
```
learning_rate = 0.1
epochs = 100
n_class = 2
rc = 0.01
Training Accuracy : 92.134831 | Validation Accuracy: 84.080717
```

5. Trial 5: Reduce the regularization constant
```
learning_rate = 0.1
epochs = 100
n_class = 2
rc = 0.001
Training Accuracy : 91.910112 | Validation Accuracy: 83.632287
```

6. Trial 6: Reduce the regularization constant
```
learning_rate = 0.1
epochs = 100
n_class = 2
rc = 0.0001
Training Accuracy : 91.629213 | Validation Accuracy: 83.856502
```

6. Trial 7: Reduce the learning rate by a factor of 5 at every 50 epoch
```
learning_rate = 0.1
epochs = 100
n_class = 2
rc = 0.01
SVM Training Accuracy : 88.539326 | Validation Accuracy: 86.098655
Note: Reduced overfitting
```

# Softmax

## All training data results

1. Trial 1
```
learning_rate = 0.01
epochs = 100
n_class = 2
rc = 0.1
accuracy: 73.391544
```

## Training + Validation test results
1. Trial 1: Initial
```
learning_rate = 0.01
epochs = 100
n_class = 2
rc = 0.1
Sotfmax Training Accuracy : 89.943820 | Validation Accuracy: 88.565022
Note: Loss increased after about 60 epochs
```

2. Trial 2: Reduce the regularization constant
```
learning_rate = 0.01
epochs = 100
n_class = 2
rc = 0.01
Sotfmax Training Accuracy : 89.550562 | Validation Accuracy: 89.461883
Note: Removed the increase in loss during training. Loss decreased continually
```
3. Trial 3: Increase the learning rate
```
learning_rate = 0.1
epochs = 100
n_class = 2
rc = 0.01
Sotfmax Training Accuracy : 90.449438 | Validation Accuracy: 86.322870
Note: The loss decreased sharply and started increasing again
```

4. Trial 4: Reduce learning rate
```
learning_rate = 0.001
epochs = 100
n_class = 2
rc = 0.01
Sotfmax Training Accuracy : 85.280899 | Validation Accuracy: 87.219731
Note: Lower accuracies. The loss function took longer to converge
```

5. Trial 5: Implement an adam variant
```
learning_rate = 0.01
epochs = 200
n_class = 2
rc = 0.1
Sotfmax Training Accuracy : 17.528090 | Validation Accuracy: 13.677130
Note: Error in the adam implementation?
```

6. Trial 7: Reduce the learning rate by a factor of 5 at every 50 epoch
```
learning_rate = 0.1
epochs = 100
n_class = 2
rc = 0.01
Sotfmax Training Accuracy : 89.662921 | Validation Accuracy: 89.237668
Note: Reduced overfitting
```