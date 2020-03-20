import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class FCN(BaseModel):
    def __init__(self, n_class=21):
        super().__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
 

    def forward(self, x):
        o = F.relu(self.conv1_1(x))
        o = F.relu(self.conv1_2(o))
        o = self.pool1(o)

        o = F.relu(self.conv2_1(o))
        o = F.relu(self.conv2_2(o))
        o = self.pool2(o)

        o = F.relu(self.conv3_1(o))
        o = F.relu(self.conv3_2(o))
        o = F.relu(self.conv3_3(o))
        o = self.pool3(o)
        pool3 = o

        o = F.relu(self.conv4_1(o))
        o = F.relu(self.conv4_2(o))
        o - F.relu(self.conv4_3(o))
        o = self.pool4(o)
        pool4 = o

        o = F.relu(self.conv5_1(o))
        o = F.relu(self.conv5_2(o))
        o = F.relu(self.conv5_3(o))
        o = self.pool5(o)
        
        o = F.relu(self.fc6(o))
        o = self.drop6(o)

        o = F.relu(self.fc7(o))
        o = self.drop7(o)

        o = self.score_fr(o)
        o = self.upscore2(o)
        upscore2 = o

        o = self.score_pool4(pool4)
        o = o[:, :, 5:(5 + upscore2.size()[2]), 5:(5 + upscore2.size()[3])]
        score_pool4c = o

        o = upscore2 + score_pool4c  
        o = self.upscore_pool4(o)
        upscore_pool4 = o  

        o = self.score_pool3(pool3)
        o = o[:, :, 9:(9 + upscore_pool4.size()[2]), 9:(9 + upscore_pool4.size()[3])]
        score_pool3c = o

        o = upscore_pool4 + score_pool3c  

        o = self.upscore8(o)
        o = o[:, :, 31:(31 + x.size()[2]), 31:(31 + x.size()[3])].contiguous()

        return o
