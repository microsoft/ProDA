# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class FCDiscriminator(nn.Module):
    """
    inplanes, planes. Patch-gan
    """

    def __init__(self, inplanes, planes = 64):
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(planes*4, planes*8, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.classifier = nn.Conv2d(planes*8, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

class FCDiscriminator_low(nn.Module):
    """
    inplanes, planes. Patch-gan
    """

    def __init__(self, inplanes, planes = 64):
        super(FCDiscriminator_low, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.classifier = nn.Conv2d(planes*4, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

class FCDiscriminator_out(nn.Module):
    """
    inplanes, planes. Patch-gan
    """

    def __init__(self, inplanes, planes = 64):
        super(FCDiscriminator_out, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(planes*4, planes*8, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.classifier = nn.Conv2d(planes*8, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

class FCDiscriminator_class(nn.Module): #TODO: whether reduce channels before pooling, whether update pred, more complex discriminator
                                        #TODO: 19 different discriminators or 1 discriminator after projection
    """
    inplanes, planes. gan
    """
    class DISCRIMINATOR(nn.Module):
        def __init__(self,inplanes, planes=64):
            super(FCDiscriminator_class.DISCRIMINATOR, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(planes*4, planes*8, kernel_size=3, stride=2, padding=1)
            self.relu = nn.ReLU(inplace=True)
            self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.classifier = nn.Conv2d(planes*8, 1, kernel_size=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.leaky_relu(x)
            x = self.classifier(x)
            return x
            
    def __init__(self, inplanes, midplanes, planes = 32):
        '''
        midplanes: channel size after reduction
        '''

        super(FCDiscriminator_class, self).__init__()
        self.inplace = inplanes
        self.midplanes = midplanes
        self.planes = planes
        self.source_unique = []
        self.target_unique = []
        self.common_unique = []
        self.discriminator = self.DISCRIMINATOR(inplanes)

        # self.fc1 = []
        # for i in range(2):
        #     self.fc1.append(nn.Linear(planes * 3, planes))
        # self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.2)
        # # self.fc2 = nn.Linear(planes, 1)
        # self.fc2 = []
        # for i in range(2):
        #     self.fc2.append(nn.Linear(planes, 1))

        # self.valid_unique= []
        # self.projection = []
        # self.projection_1 = []
        # for i in range(19):
        #     self.projection.append(nn.Conv2d(inplanes, planes*3, kernel_size=1))
        # self.projection = nn.ModuleList(self.projection)
        # self.fc1 = nn.ModuleList(self.fc1)
        # self.fc2 = nn.ModuleList(self.fc2)

    def forward(self, x):
        x = self.discriminator(x)
        pass
        return x

    # def forward(self, vectors, ids): #TODO: code when batch>1, default batch size is 1 
    #     # pred = output_softmax.data.max(1)[1].cpu().numpy()
    #     # unique, counts = np.unique(pred, return_counts=True)
    #     # output_softmax = output_softmax.detach()
    #     flag = True
    #     # pred_pooling = F.adaptive_avg_pool2d(output_softmax, 1)
    #     # t = -1
    #     s_staff = [0,1,2,4,5,6,8,9,10,11,12,13,14] #3 4 6 11 12
    #     s_object = [4,5,6,7,11,12,13,14,15,16,17,18]
    #     loss_weight = []
    #     output = torch.FloatTensor(1).fill_(0)
    #     for i in range(len(ids)):
    #         # if counts[t] < 30:
    #         #     continue
    #         if ids[i] not in self.valid_unique:
    #             continue
    #         out = vectors[i].reshape([1, -1, 1, 1])
    #         out = self.projection[0](out)
    #         if i in s_staff:
    #             _u = 0
    #         else:
    #             continue
    #         out = self.relu(out)
    #         out = self.fc1[_u](out.view(1, -1))
    #         out = self.dropout(out)
    #         out = self.fc2[_u](out)

    #         if not flag:
    #             output = torch.cat((output, out), dim=0)
    #         else:
    #             output = out
    #             flag = False
    #         loss_weight.append(1)
    #     return output, loss_weight

    def calc_common_unique(self, source_unique, target_unique):
        self.source_unique = source_unique
        self.target_unique = target_unique
        self.common_unique = []
        for i in range(19):
            if (i in self.source_unique) and (i in self.target_unique):
                self.common_unique.append(i)
        pass

    def calc_valid_unique(self, classes_list):
        self.valid_unique = []
        for i in range(19):
            if (i in classes_list):
                self.valid_unique.append(i)