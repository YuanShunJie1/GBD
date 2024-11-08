"""
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import weights_init#,  BasicBlock
import torch
# from models.mixtext import MixText
# utils/utils.py
from functools import partial


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out




class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        activation1 = out
        out = self.layer2(out)
        activation2 = out
        out = self.layer3(out)
        activation3 = out
        out = self.layer4(out)
        
        out = self.avgpool(out)
        hidden = out.view(out.size(0), -1)
        out = self.linear(hidden)
        return out



def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(14 * 28, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = x.reshape(-1, 14 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BottomModelForMnist(nn.Module):
    def __init__(self):
        super(BottomModelForMnist, self).__init__()
        self.mlpnet = MLPNet()

    def forward(self, x):
        x = self.mlpnet(x)
        return x

class TopModelForMnist(nn.Module):
    def __init__(self, input_size, output_size):
        super(TopModelForMnist, self).__init__()
        self.fc1top = nn.Linear(input_size, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, output_size)
        self.bn0top = nn.BatchNorm1d(input_size)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)
    
    def forward(self, x):
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)




class FC1(nn.Module):
    def __init__(self, num_passive, division):
        super(FC1, self).__init__()
        self.num_passive = num_passive

        self.passive = nn.ModuleList([
            BottomModelForMnist()
        ])

        self.active = TopModelForMnist()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        data = list(data)
        emb = []
        for i in range(self.num_passive):
            tmp_emb = self.passive[i](data[i])
            emb.append(tmp_emb)

        agg_emb = torch.cat(emb, dim=1)
        logit = self.active(agg_emb)
        pred = self.softmax(logit)
        return emb, logit, pred


class BottomModelForCifar10(nn.Module):
    def __init__(self):
        super(BottomModelForCifar10, self).__init__()
        self.resnet20 = resnet18(num_classes=128)

    def forward(self, x):
        x = self.resnet20(x)
        # x = F.normalize(x, dim=1)
        return x

class TopModelForCifar10(nn.Module):
    def __init__(self, input_size, output_size):
        super(TopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(input_size, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, output_size)
        self.bn0top = nn.BatchNorm1d(input_size)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)
    
    def forward(self, x):
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return x
        # return F.log_softmax(x, dim=1)



class ResNetForCIFAR(nn.Module):
    def __init__(self, num_classes, num_passive):
        super(ResNetForCIFAR, self).__init__()
        self.num_passive = num_passive

        self.passive = nn.ModuleList([
            BottomModelForCifar10() for _ in range(num_passive)
        ])
        self.active = TopModelForCifar10(input_size=128 * num_passive, output_size=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = list(x)
        emb = []
        for i in range(self.num_passive):
            emb.append(self.passive[i](x[i]))
                
        agg_emb = torch.cat(emb, dim=1)
        logit = self.active(agg_emb)
        pred = self.softmax(logit)

        return emb, logit, pred
    
    def _aggregate(self, x):
        agg_emb = torch.cat(x, dim=1)
        agg_emb = agg_emb.view(agg_emb.size(0), -1)
        return agg_emb


entire = {
    'mnist': FC1,
    'fashionmnist': FC1,
    'cifar10': partial(ResNetForCIFAR, num_classes=10),
    'cifar100': partial(ResNetForCIFAR , num_classes=100),
    # 'criteo': partial(DeepFM, feature_sizes=feature_sizes, emb_size=4, hidden_size=32, dropout=0.5, num_classes=2),
    'cinic10': partial(ResNetForCIFAR , num_classes=10)
}


# class BottomModelForCinic10(nn.Module):
#     def __init__(self):
#         super(BottomModelForCinic10, self).__init__()
#         self.resnet20 = resnet20(num_classes=10)

#     def forward(self, x):
#         x = self.resnet20(x)
#         x = F.normalize(x, dim=1)
#         return x


# class TopModelForCinic10(nn.Module):
#     def __init__(self):
#         super(TopModelForCinic10, self).__init__()
#         self.fc1top = nn.Linear(10*2, 20)
#         self.fc2top = nn.Linear(20, 10)
#         self.fc3top = nn.Linear(10, 10)
#         self.fc4top = nn.Linear(10, 10)
#         self.bn0top = nn.BatchNorm1d(10*2)
#         self.bn1top = nn.BatchNorm1d(20)
#         self.bn2top = nn.BatchNorm1d(10)
#         self.bn3top = nn.BatchNorm1d(10)

#         self.apply(weights_init)

#     def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
#         output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
#         x = output_bottom_models
#         x = self.fc1top(F.relu(self.bn0top(x)))
#         x = self.fc2top(F.relu(self.bn1top(x)))
#         x = self.fc3top(F.relu(self.bn2top(x)))
#         x = self.fc4top(F.relu(self.bn3top(x)))
#         return F.log_softmax(x, dim=1)




# class BottomModelForCifar100(nn.Module):
#     def __init__(self):
#         super(BottomModelForCifar100, self).__init__()
#         self.resnet20 = resnet20(num_classes=100)

#     def forward(self, x):
#         x = self.resnet20(x)
#         x = F.normalize(x, dim=1)
#         return x


# class TopModelForCifar100(nn.Module):
#     def __init__(self):
#         super(TopModelForCifar100, self).__init__()
#         self.fc1top = nn.Linear(100*2, 200)
#         self.fc2top = nn.Linear(200, 100)
#         self.fc3top = nn.Linear(100, 100)
#         self.fc4top = nn.Linear(100, 100)
#         self.bn0top = nn.BatchNorm1d(100*2)
#         self.bn1top = nn.BatchNorm1d(200)
#         self.bn2top = nn.BatchNorm1d(100)
#         self.bn3top = nn.BatchNorm1d(100)

#         self.apply(weights_init)

#     def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
#         output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
#         x = output_bottom_models
#         x = self.fc1top(F.relu(self.bn0top(x)))
#         x = self.fc2top(F.relu(self.bn1top(x)))
#         x = self.fc3top(F.relu(self.bn2top(x)))
#         x = self.fc4top(F.relu(self.bn3top(x)))
#         return F.log_softmax(x, dim=1)




# class BottomModelForTinyImageNet(nn.Module):
#     def __init__(self):
#         super(BottomModelForTinyImageNet, self).__init__()
#         self.resnet56 = resnet56(num_classes=200)

#     def forward(self, x):
#         x = self.resnet56(x)
#         x = F.normalize(x, dim=1)
#         return x


# class TopModelForTinyImageNet(nn.Module):
#     def __init__(self):
#         super(TopModelForTinyImageNet, self).__init__()
#         self.fc1top = nn.Linear(400, 400)
#         self.fc2top = nn.Linear(400, 200)
#         self.fc3top = nn.Linear(200, 200)
#         self.bn0top = nn.BatchNorm1d(400)
#         self.bn1top = nn.BatchNorm1d(400)
#         self.bn2top = nn.BatchNorm1d(200)

#         self.apply(weights_init)

#     def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
#         output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
#         x = output_bottom_models
#         x = self.fc1top(F.relu(self.bn0top(x)))
#         x = self.fc2top(F.relu(self.bn1top(x)))
#         x = self.fc3top(F.relu(self.bn2top(x)))
#         return F.log_softmax(x, dim=1)





# class BottomModelForImageNet(nn.Module):
#     def __init__(self):
#         super(BottomModelForImageNet, self).__init__()
#         self.resnet56 = resnet56(num_classes=1000)

#     def forward(self, x):
#         x = self.resnet56(x)
#         x = F.normalize(x, dim=1)
#         return x


# class TopModelForImageNet(nn.Module):
#     def __init__(self):
#         super(TopModelForImageNet, self).__init__()
#         self.fc1top = nn.Linear(2000, 2000)
#         self.fc2top = nn.Linear(2000, 1000)
#         self.fc3top = nn.Linear(1000, 1000)
#         self.bn0top = nn.BatchNorm1d(2000)
#         self.bn1top = nn.BatchNorm1d(2000)
#         self.bn2top = nn.BatchNorm1d(1000)

#         self.apply(weights_init)

#     def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
#         output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
#         x = output_bottom_models
#         x = self.fc1top(F.relu(self.bn0top(x)))
#         x = self.fc2top(F.relu(self.bn1top(x)))
#         x = self.fc3top(F.relu(self.bn2top(x)))
#         return F.log_softmax(x, dim=1)



# class TopModelForYahoo(nn.Module):

#     def __init__(self):
#         super(TopModelForYahoo, self).__init__()
#         self.fc1_top = nn.Linear(20, 10)
#         self.fc2_top = nn.Linear(10, 10)
#         self.fc3_top = nn.Linear(10, 10)
#         self.fc4_top = nn.Linear(10, 10)
#         self.bn0top = nn.BatchNorm1d(20)
#         self.bn1top = nn.BatchNorm1d(10)
#         self.bn2top = nn.BatchNorm1d(10)
#         self.bn3top = nn.BatchNorm1d(10)
#         self.apply(weights_init)

#     def forward(self,input_tensor_top_model_a, input_tensor_top_model_b):
#         output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
#         x = self.bn0top(output_bottom_models)
#         x = F.relu(x)
#         x = self.fc1_top(x)
#         x = self.bn1top(x)
#         x = F.relu(x)
#         x = self.fc2_top(x)
#         x = self.bn2top(x)
#         x = F.relu(x)
#         x = self.fc3_top(x)
#         x = self.bn3top(x)
#         x = F.relu(x)
#         x = self.fc4_top(x)

#         return x




# # class BottomModelForYahoo(nn.Module):

# #     def __init__(self, n_labels):
# #         super(BottomModelForYahoo, self).__init__()
# #         self.mixtext_model = MixText(n_labels, True)

# #     def forward(self, x):
# #         x = self.mixtext_model(x)
# #         x = F.normalize(x, dim=1)
# #         return x


# D_ = 2 ** 13


# class TopModelForCriteo(nn.Module):

#     def __init__(self):
#         super(TopModelForCriteo, self).__init__()
#         self.fc1_top = nn.Linear(8, 8)
#         self.fc2_top = nn.Linear(8, 4)
#         self.fc3_top = nn.Linear(4, 2)
#         self.apply(weights_init)

#     def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
#         output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
#         x = output_bottom_models
#         x = F.relu(x)
#         x = self.fc1_top(x)
#         x = F.relu(x)
#         x = self.fc2_top(x)
#         x = F.relu(x)
#         x = self.fc3_top(x)
#         return x






# class BottomModelForCriteo(nn.Module):

#     def __init__(self, half=14, is_adversary=False):
#         super(BottomModelForCriteo, self).__init__()
#         if not is_adversary:
#             half = D_ - half
#         self.fc1 = nn.Linear(half, 64)
#         self.fc2 = nn.Linear(64, 16)
#         self.fc3 = nn.Linear(16, 4)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         return x


# # class TopModelForBcw(nn.Module):
# #     def __init__(self):
# #         super(TopModelForBcw, self).__init__()
# #         self.fc1_top = nn.Linear(4, 4)
# #         self.bn0_top = nn.BatchNorm1d(4)
# #         self.fc2_top = nn.Linear(4, 4)
# #         self.bn1_top = nn.BatchNorm1d(4)
# #         self.fc3_top = nn.Linear(4, 4)
# #         self.bn2_top = nn.BatchNorm1d(4)
# #         self.fc4_top = nn.Linear(4, 4)
# #         self.bn3_top = nn.BatchNorm1d(4)
# #         self.apply(weights_init)

# #     def forward(self,input_tensor_top_model_a, input_tensor_top_model_b):
# #         output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
# #         x = self.bn0_top(output_bottom_models)
# #         x = F.relu(x)
# #         x = self.fc1_top(x)
# #         x = self.bn1_top(x)
# #         x = F.relu(x)
# #         x = self.fc2_top(x)
# #         x = self.bn2_top(x)
# #         x = F.relu(x)
# #         x = self.fc3_top(x)
# #         x = self.bn3_top(x)
# #         x = F.relu(x)        
# #         x = self.fc4_top(x)
# #         return x


# class TopModelForBcw(nn.Module):
#     def __init__(self,):
#         super(TopModelForBcw, self).__init__()
#         self.fc1_top = nn.Linear(20, 20)
#         self.bn0_top = nn.BatchNorm1d(20)
#         self.fc2_top = nn.Linear(20, 20)
#         self.bn1_top = nn.BatchNorm1d(20)
#         self.fc3_top = nn.Linear(20, 2)
#         self.bn2_top = nn.BatchNorm1d(20)
#         self.apply(weights_init)

#     def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
#         output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
#         x = self.bn0_top(output_bottom_models)
#         x = F.relu(x)
#         x = self.fc1_top(x)
#         x = self.bn1_top(x)
#         x = F.relu(x)
#         x = self.fc2_top(x)
#         x = self.bn2_top(x)
#         x = F.relu(x)
#         x = self.fc3_top(x)
#         return x





# class BottomModelForBcw(nn.Module):
#     def __init__(self, half=14, is_adversary=False):
#         super(BottomModelForBcw, self).__init__()
#         if not is_adversary:
#             half = 28 - half
#         self.fc1 = nn.Linear(half, 20)
#         self.fc2 = nn.Linear(20, 20)
#         self.fc3 = nn.Linear(20, 10)
#         self.bn1 = nn.BatchNorm1d(20)
#         self.bn2 = nn.BatchNorm1d(20)
#         self.apply(weights_init)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         return x



