from torch.autograd import Function
import torch.nn as nn
import torch
import torch.nn.functional as F

class ReverseLayerFunction(Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None

class Feature_Extractor(nn.Module):

    def __init__(self):
        super(Feature_Extractor, self).__init__()
        self.model = nn.Sequential(
            # state = (3,28,28)
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            # state = (64,12,12)
            nn.Conv2d(64, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            # state = (64,4,4)
            nn.Conv2d(64, 128, kernel_size= 5, padding= 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d()
        )

    def forward(self, input):
        input = input.expand(input.data.shape[0], 3, 28, 28)
        x = self.model(input)
        output = x.view(-1, 128 * 4 * 4)
        return output

class Label_Predictor(nn.Module):

    def __init__(self):
        super(Label_Predictor, self).__init__()
        self.model = nn.Sequential(
            # state = (*,48 * 4 * 4)
            nn.Linear(128 * 4 * 4, 4096),
            # state = (*,100)
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),
            # state = (*,100)
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 10),
            # state = (*,10)
            nn.LogSoftmax(dim = 1)
        )

    def forward(self, input):
        output = self.model(input)
        return output

class Domain_classifier(nn.Module):

    def __init__(self):
        super(Domain_classifier, self).__init__()
        self.model = nn.Sequential(
            # state = (*,48 * 4 * 4)
            nn.Linear(128 * 4 * 4, 1024),
            # state = (*,100)
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            # state = (*,100)
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(1024, 1),
            # state = (*,2)
            # nn.LogSoftmax(dim = 1)
        )

    def forward(self, input, lambda_):
        input = ReverseLayerFunction.apply(input, lambda_)
        output = self.model(input)
        output = torch.sigmoid(output)
        return output
