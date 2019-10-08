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
            nn.Conv2d(3, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # state = (32,12,12)
            nn.Conv2d(32, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            # state = (48,4,4)
            nn.ReLU(True)
        )

    def forward(self, input):
        input = input.expand(input.data.shape[0], 3, 28, 28)
        x = self.model(input)
        output = x.view(-1, 48 * 4 * 4)
        return output

class Label_Predictor(nn.Module):

    def __init__(self):
        super(Label_Predictor, self).__init__()
        self.model = nn.Sequential(
            # state = (*,48 * 4 * 4)
            nn.Linear(48 * 4 * 4, 100),
            # state = (*,100)
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(),
            # state = (*,100)
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10),
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
            nn.Linear(48 * 4 * 4, 100),
            # state = (*,100)
            # nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 1),
            # state = (*,2)
            # nn.LogSoftmax(dim = 1)
        )

    def forward(self, input, lambda_):
        input = ReverseLayerFunction.apply(input, lambda_)
        output = self.model(input)
        output = torch.sigmoid(output)
        return output
