import torch
import torch.nn as nn
import torch.nn.init as init
from functools import partial
from utils.datasets import feature_sizes
import torch.nn.functional as F
from attackers.ubd import UBDDefense


torch.cuda.set_device(0)
device = torch.device(f'cuda:{0}')

def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)



class PadAndSubsample(nn.Module):
    def __init__(self, planes):
        super(PadAndSubsample, self).__init__()
        self.planes = planes

    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes // 4, self.planes // 4), "constant", 0)

class BasicBlock_L(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_L, self).__init__()
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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                # self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
                self.shortcut = PadAndSubsample(planes)
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, kernel_size, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], kernel_size, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # [bs,3,32,16]
        out = F.relu(self.bn1(self.conv1(x)))
        # [bs,16,32,16]
        out = self.layer1(out)
        # [bs,16,32,16]
        out = self.layer2(out)
        # [bs,32,16,8]
        out = self.layer3(out)
        # [bs,64,8,4]
        out = F.avg_pool2d(out, out.size()[2:])
        # [bs,64,1,1]
        out = out.view(out.size(0), -1)
        # [bs,64]
        out = self.linear(out)
        # [bs,10]
        return out


def resnet20(kernel_size=(3, 3), num_classes=12):
    return ResNet(block=BasicBlock, num_blocks=[3, 3, 3], kernel_size=kernel_size, num_classes=num_classes)


class ResNet_M(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_M, self).__init__()
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

    def forward(self, x, return_hidden=False, return_activation=False):
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

        if return_hidden:
            return out, hidden
        elif return_activation:  # for NAD
            return out, activation1, activation2, activation3
        else:
            return out

def resnet34(num_classes=12):
    return ResNet_M(BasicBlock_L, [3, 4, 6, 3], num_classes=num_classes)

def weights_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

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
    def __init__(self):
        super(TopModelForMnist, self).__init__()
        self.fc1top = nn.Linear(128*2, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(128*2)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)
    
    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b, agged_inputs=None, agged=False):
        if agged:
            
            # den = 1              
            # noise = torch.randn_like(agged_inputs) / den
            # agged_inputs = agged_inputs + noise
            
            x = self.fc1top(F.relu(self.bn0top(agged_inputs)))
            x = self.fc2top(F.relu(self.bn1top(x)))
            x = self.fc3top(F.relu(self.bn2top(x)))
            x = self.fc4top(F.relu(self.bn3top(x)))
        else:
            output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
            
            # den = 1              
            # noise = torch.randn_like(output_bottom_models) / den
            # output_bottom_models = output_bottom_models + noise
            
            x = output_bottom_models
            x = self.fc1top(F.relu(self.bn0top(x)))
            x = self.fc2top(F.relu(self.bn1top(x)))
            x = self.fc3top(F.relu(self.bn2top(x)))
            x = self.fc4top(F.relu(self.bn3top(x)))           
        return x

class FC1(nn.Module):
    def __init__(self, num_passive):
        super(FC1, self).__init__()
        self.num_passive = num_passive
        self.passive = nn.ModuleList([
            BottomModelForMnist().cuda() for i in range(num_passive)
        ])
        self.active = TopModelForMnist().cuda()

    def forward(self, data):
        emb = []
        for i in range(self.num_passive):
            tmp_emb = self.passive[i](data[i])
            emb.append(tmp_emb)

        agg_emb = torch.cat(emb, dim=1)
        
        logit = self.active(None, None, agged_inputs=agg_emb, agged=True)
        return logit

    def _aggregate(self, emb):
        agg_emb = torch.cat(emb, dim=1)
        return agg_emb





class BottomModelForCinic10(nn.Module):
    def __init__(self):
        super(BottomModelForCinic10, self).__init__()
        self.resnet20 = resnet20(num_classes=10)

    def forward(self, x):
        x = self.resnet20(x)
        return x

class TopModelForCinic10(nn.Module):
    def __init__(self):
        super(TopModelForCinic10, self).__init__()
        self.fc1top = nn.Linear(20, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b, agged_inputs=None, agged=False):
        if agged:
            x = self.fc1top(F.relu(self.bn0top(agged_inputs)))
            x = self.fc2top(F.relu(self.bn1top(x)))
            x = self.fc3top(F.relu(self.bn2top(x)))
            x = self.fc4top(F.relu(self.bn3top(x)))
        else:
            output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
            x = output_bottom_models
            x = self.fc1top(F.relu(self.bn0top(x)))
            x = self.fc2top(F.relu(self.bn1top(x)))
            x = self.fc3top(F.relu(self.bn2top(x)))
            x = self.fc4top(F.relu(self.bn3top(x)))
        return x


class FC_CIFAR10(nn.Module):
    def __init__(self, num_passive):
        super(FC_CIFAR10, self).__init__()
        self.num_passive = num_passive
        self.passive = nn.ModuleList([
            BottomModelForCifar10().cuda() for i in range(num_passive)
        ])
        self.active = TopModelForCifar10().cuda()

    def forward(self, data):
        emb = []
        for i in range(self.num_passive):
            tmp_emb = self.passive[i](data[i])
            emb.append(tmp_emb)

        agg_emb = torch.cat(emb, dim=1)
        logit = self.active(None, None, agged_inputs=agg_emb, agged=True)
        return logit

    def _aggregate(self, emb):
        agg_emb = torch.cat(emb, dim=1)
        return agg_emb



class BottomModelForCifar10(nn.Module):
    def __init__(self):
        super(BottomModelForCifar10, self).__init__()
        self.resnet20 = resnet20(num_classes=10)

    def forward(self, x):
        x = self.resnet20(x)
        return x

class TopModelForCifar10(nn.Module):
    def __init__(self):
        super(TopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(20, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b, agged_inputs=None, agged=False):
        if agged:

            # den = 1              
            # noise = torch.randn_like(agged_inputs) / den
            # agged_inputs = agged_inputs + noise
            

            x = self.fc1top(F.relu(self.bn0top(agged_inputs)))
            x = self.fc2top(F.relu(self.bn1top(x)))
            x = self.fc3top(F.relu(self.bn2top(x)))
            x = self.fc4top(F.relu(self.bn3top(x)))
        else:
            output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
            
            # den = 1              
            # noise = torch.randn_like(output_bottom_models) / den
            # output_bottom_models = output_bottom_models + noise
            
            x = output_bottom_models
            x = self.fc1top(F.relu(self.bn0top(x)))
            x = self.fc2top(F.relu(self.bn1top(x)))
            x = self.fc3top(F.relu(self.bn2top(x)))
            x = self.fc4top(F.relu(self.bn3top(x)))
        return x














class BottomModelForCDC0(nn.Module):
    def __init__(self):
        super(BottomModelForCDC0, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BottomModelForCDC1(nn.Module):
    def __init__(self):
        super(BottomModelForCDC1, self).__init__()
        self.fc1 = nn.Linear(11, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TopModelForCDC(nn.Module):
    def __init__(self):
        super(TopModelForCDC, self).__init__()
        self.fc1top = nn.Linear(20, 20)
        self.fc2top = nn.Linear(20, 20)
        self.fc3top = nn.Linear(20, 21)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(20)
        self.apply(weights_init)
    
    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b, agged_inputs=None, agged=False):
        if agged:
            x = self.fc1top(F.relu(self.bn0top(agged_inputs)))
            x = self.fc2top(F.relu(self.bn1top(x)))
            x = self.fc3top(F.relu(self.bn2top(x)))
        else:
            output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
            x = output_bottom_models
            x = self.fc1top(F.relu(self.bn0top(x)))
            x = self.fc2top(F.relu(self.bn1top(x)))
            x = self.fc3top(F.relu(self.bn2top(x)))
        return x

class FC_CDC(nn.Module):
    def __init__(self, num_passive):
        super(FC_CDC, self).__init__()
        self.num_passive = num_passive
        self.passive = nn.ModuleList([
            BottomModelForCDC0().cuda(),
            BottomModelForCDC1().cuda(),           
        ])
        self.active = TopModelForCDC().cuda()

    def forward(self, data):
        emb = []
        for i in range(self.num_passive):
            tmp_emb = self.passive[i](data[i])
            emb.append(tmp_emb)

        agg_emb = torch.cat(emb, dim=1)
        logit = self.active(None, None, agged_inputs=agg_emb, agged=True)
        return logit

    def _aggregate(self, emb):
        agg_emb = torch.cat(emb, dim=1)
        return agg_emb






class BottomModelForAIDS0(nn.Module):
    def __init__(self):
        super(BottomModelForAIDS0, self).__init__()
        self.fc1 = nn.Linear(12, 20)
        self.fc2 = nn.Linear(20, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BottomModelForAIDS1(nn.Module):
    def __init__(self):
        super(BottomModelForAIDS1, self).__init__()
        self.fc1 = nn.Linear(11, 20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TopModelForAIDS(nn.Module):
    def __init__(self):
        super(TopModelForAIDS, self).__init__()
        self.fc1top = nn.Linear(20, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 21)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.apply(weights_init)
    
    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b, agged_inputs=None, agged=False):
        if agged:
            x = self.fc1top(F.relu(self.bn0top(agged_inputs)))
            x = self.fc2top(F.relu(self.bn1top(x)))
            x = self.fc3top(F.relu(self.bn2top(x)))
        else:
            output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
            x = output_bottom_models
            x = self.fc1top(F.relu(self.bn0top(x)))
            x = self.fc2top(F.relu(self.bn1top(x)))
            x = self.fc3top(F.relu(self.bn2top(x)))
        return x

class FC_AIDS(nn.Module):
    def __init__(self, num_passive):
        super(FC_AIDS, self).__init__()
        self.num_passive = num_passive
        self.passive = nn.ModuleList([
            BottomModelForAIDS0().cuda(),
            BottomModelForAIDS1().cuda(),           
        ])
        self.active = TopModelForAIDS().cuda()

    def forward(self, data):
        emb = []
        for i in range(self.num_passive):
            tmp_emb = self.passive[i](data[i])
            emb.append(tmp_emb)

        agg_emb = torch.cat(emb, dim=1)
        logit = self.active(None, None, agged_inputs=agg_emb, agged=True)
        return logit

    def _aggregate(self, emb):
        agg_emb = torch.cat(emb, dim=1)
        return agg_emb

















class FC_ImageNet12(nn.Module):
    def __init__(self, num_passive):
        super(FC_ImageNet12, self).__init__()
        self.num_passive = num_passive
        self.passive = nn.ModuleList([
            BottomModelForImageNet12().cuda() for i in range(num_passive)
        ])
        self.active = TopModelForImageNet12().cuda()

    def forward(self, data):
        emb = []
        for i in range(self.num_passive):
            tmp_emb = self.passive[i](data[i])
            emb.append(tmp_emb)

        agg_emb = torch.cat(emb, dim=1)
        logit = self.active(None, None, agged_inputs=agg_emb, agged=True)
        return logit

    def _aggregate(self, emb):
        agg_emb = torch.cat(emb, dim=1)
        return agg_emb



class BottomModelForImageNet12(nn.Module):
    def __init__(self):
        super(BottomModelForImageNet12, self).__init__()
        self.resnet34 = resnet34(num_classes=128)

    def forward(self, x):
        x = self.resnet34(x)
        return x

class TopModelForImageNet12(nn.Module):
    def __init__(self):
        super(TopModelForImageNet12, self).__init__()
        self.fc1top = nn.Linear(128*2, 128*2)
        self.fc2top = nn.Linear(128*2, 128)
        self.fc3top = nn.Linear(128, 12)
        self.bn0top = nn.BatchNorm1d(128*2)
        self.bn1top = nn.BatchNorm1d(128*2)
        self.bn2top = nn.BatchNorm1d(128)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b, agged_inputs=None, agged=False):
        if agged:
            x = self.fc1top(F.relu(self.bn0top(agged_inputs)))
            x = self.fc2top(F.relu(self.bn1top(x)))
            x = self.fc3top(F.relu(self.bn2top(x)))
        else:
            output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
            x = output_bottom_models
            x = self.fc1top(F.relu(self.bn0top(x)))
            x = self.fc2top(F.relu(self.bn1top(x)))
            x = self.fc3top(F.relu(self.bn2top(x)))
        return x





entire = {
    'mnist'       : FC1,
    'fashionmnist': FC1,
    'cifar10'     :FC_CIFAR10,
    'cinic10'     :FC_CIFAR10,
    'cdc'         :FC_CDC,
    'aids'        :FC_AIDS,
    'imagenet12': FC_ImageNet12,
    # 'cifar10': partial(ResNet, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=10),
    # 'cifar100': partial(ResNet, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=100),
    # 'criteo': partial(DeepFM, feature_sizes=feature_sizes, emb_size=4, hidden_size=32, dropout=0.5, num_classes=2),
    # 'cinic10': partial(ResNet, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=10)
}
