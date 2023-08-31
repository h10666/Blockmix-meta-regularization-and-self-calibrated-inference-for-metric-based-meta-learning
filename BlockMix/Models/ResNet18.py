import torch
import torch.nn as nn
import torchvision.models as models
import math
import numpy as np

SEED = 3
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Net(nn.Module):
    def __init__(self, num_classes=80, norm=True, scale=True):
        super(Net,self).__init__()
        self.extractor = models.resnet18(pretrained=False)
        self.feature_dim = self.extractor.fc.in_features
        self.extractor.fc = Identity()
        self.classifier = Classifier(self.feature_dim, num_classes)
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.norm = norm
        self.scale = scale

    def forward(self, x):
        x = self.extractor(x)
        feature = self.l2_norm(x)
        score = self.classifier(feature*self.s)
        return feature, score

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def weight_norm(self):
        w = self.classifier.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier.fc.weight.data = w.div(norm.expand_as(w))

class Classifier(nn.Module):
    def __init__(self,input_dim, num_classes):
        super(Classifier,self).__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x


# if __name__ == '__main__':
#     data = torch.ones((64,3,224,224))
#     model = Net()
#     out = model(data)
#     print(out[0].shape)

