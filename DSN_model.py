import torch.nn as nn
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_

        return output, None

class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.interp(x, scale_factor = self.scale_factor)
        return x

class DSN(nn.Module):
    def __init__(self, feature_size = 100, num_class = 10):
        super(DSN, self).__init__()
        self.feature_size = feature_size

        # private target encoder

        self.target_encoder_conv = nn.Sequential(
            # state = (3,28,28)
            nn.Conv2d( 3, 32, kernel_size=5,padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d( 32, 64, kernel_size=5,padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # state = (64,7,7)
        )

        self.target_encoder_fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, feature_size),
            nn.ReLU(True)
        )

        # private source encoder

        self.source_encoder_conv = nn.Sequential(
            # state = (3,28,28)
            nn.Conv2d( 3, 32, kernel_size=5,padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d( 32, 64, kernel_size=5,padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # state = (64,7,7)
        )

        self.source_encoder_fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, feature_size),
            nn.ReLU(True)
        )

        # shared encoder

        self.shared_encoder_conv = nn.Sequential(
            # state = (3,28,28)
            nn.Conv2d( 3, 32, kernel_size=5,padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d( 32, 48, kernel_size=5,padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # state = (64,7,7)
        )

        self.shared_encoder_fc = nn.Sequential(
            nn.Linear(7 * 7 * 48, feature_size),
            nn.ReLU(True)
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 100),
            nn.ReLU(True),
            nn.Linear(100, num_class)
        )

        self.shared_domain_classifier = nn.Sequential(
            # state = (100,1,1)
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 1)
        )

        # shared decoder (small decoder)

        self.shared_decoder_fc = nn.Sequential(
            nn.Linear(feature_size, 588),
            nn.ReLU(True),

        )

        self.shared_decoder_conv = nn.Sequential(
            # state = (3,14,14)
            nn.Conv2d( 3, 16, kernel_size=5,padding=2),
            nn.ReLU(),
            # state = (16,14,14)
            nn.Conv2d( 16, 16, kernel_size=5,padding=2),
            nn.ReLU(),
            Interpolate(scale_factor=2),
            # state = (16,28,28)
            nn.Conv2d( 16, 16, kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d( 16, 3, kernel_size=3,padding=1)
            # state = (3,28,28)
        )

    def forward(self, input_data, mode, recon_type, lambda_ = 0.0):

        result = []

        # private encoder
        if mode == 'source':
            private_tmp = self.source_encoder_conv(input_data)
            private_tmp = private_tmp.view(-1, 64 * 7 * 7)
            private_feature = self.source_encoder_fc(private_tmp)
        elif mode == 'target':
            private_tmp = self.target_encoder_conv(input_data)
            private_tmp = private_tmp.view(-1, 64 * 7 * 7)
            private_feature = self.target_encoder_fc(private_tmp)

        result.append(private_feature)

        # shared encoder
        shared_tmp = self.shared_encoder_conv(input_data)
        shared_tmp = shared_tmp.view(-1, 48 * 7 * 7)
        shared_feature = self.shared_encoder_fc(shared_tmp)
        result.append(shared_feature)

        reversed_shared_feature = ReverseLayerF.apply(shared_feature, lambda_)
        domain_label = self.shared_domain_classifier(reversed_shared_feature)
        result.append(domain_label)

        # classifier (source only)
        if mode == 'source':
            class_label = self.classifier(shared_feature)
            result.append(class_label)

        # shared decoder
        if recon_type == 'share':
            union_code = shared_feature
        elif recon_type == 'all':
            union_code = private_feature + shared_feature
        elif recon_type == 'private':
            union_code = private_feature

        rec_tmp = self.shared_decoder_fc(union_code)
        rec_tmp = rec_tmp.view(-1, 3, 14, 14)
        recon_img = self.shared_decoder_conv(rec_tmp)

        result.append(recon_img)

        return result # [private_feature,shared_feature,domain_label,class_label,recon_img]
