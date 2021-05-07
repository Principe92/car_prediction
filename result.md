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

3. Trial 3: Reduce the learning rate
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