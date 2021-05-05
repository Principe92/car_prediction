import torch
from torchvision import models


class SVM(torch.nn.Module):
    def __init__(self):
        super(SVM, self).__init__()

        self.model = models.resnet18(pretrained=True)
        

        # self.model.fc = torch.nn.Sequential(
        #     torch.nn.Linear(512, 64)
        # )

    def forward(self, x):

        return self.model(x)


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.model = models.resnet18(pretrained=True)

        # self.model.fc = torch.nn.Sequential(
        #     torch.nn.Linear(512, 64),
        #     torch.nn.Softmax()
        # )

    def forward(self, x):

        return self.model(x)
