import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################################################
# Model converted from Caffe to PyTorch using MMDnn: https://github.com/Microsoft/MMdnn 
# Model reference: https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
#
# Center crop, model trained for 60 epochs
# +---------------+---------------+
# | Top-1 acc.(%) | Top-5 acc.(%) |  
# +---------------+---------------+
# |     68.7      |     88.9      |
# +---------------+---------------+
#
##############################################################################################

__all__ = ['GoogLeNet', 'googlenet']

pretrained_settings = {
    'googlenet': {
        'imagenet': {
            # Was ported using python2 (may trigger warning)
            'url': 'https://www.dropbox.com/s/bnbj496ef1n98yi/googlenet.pth',
            'input_space': 'BGR',
            'input_size': [3, 224, 224],
            'input_range': [0, 255],
            'mean': [104, 117, 128],
            'std': [1, 1, 1],
            'num_classes': 1000
        }
    }
}


class GoogLeNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()
        
        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv1_relu_7x7 = nn.ReLU(inplace=True)
        self.pool1_3x3_s2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace=True)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_relu_3x3 = nn.ReLU(inplace=True)
        self.pool2_3x3_s2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)

        # Inception 3a
        self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_5x5_reduce = nn.Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_3x3_reduce = nn.Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_pool_proj = nn.Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_5x5 = nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.inception_3a_3x3 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Inception 3b
        self.inception_3b_3x3_reduce = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_5x5_reduce = nn.Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_pool_proj = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_5x5 = nn.Conv2d(32, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

        # Inception 4a
        self.inception_4a_1x1 = nn.Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_3x3_reduce = nn.Conv2d(480, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_5x5_reduce = nn.Conv2d(480, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_pool_proj = nn.Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_3x3 = nn.Conv2d(96, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_5x5 = nn.Conv2d(16, 48, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.inception_4b_5x5_reduce = nn.Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_1x1 = nn.Conv2d(512, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_3x3_reduce = nn.Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_pool_proj = nn.Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_5x5 = nn.Conv2d(24, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.inception_4b_3x3 = nn.Conv2d(112, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_5x5_reduce = nn.Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_1x1 = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_3x3_reduce = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_pool_proj = nn.Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_5x5 = nn.Conv2d(24, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.inception_4c_3x3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_3x3_reduce = nn.Conv2d(512, 144, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_1x1 = nn.Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_5x5_reduce = nn.Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_pool_proj = nn.Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_3x3 = nn.Conv2d(144, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_5x5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.inception_4e_5x5_reduce = nn.Conv2d(528, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_1x1 = nn.Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_3x3_reduce = nn.Conv2d(528, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_pool_proj = nn.Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_5x5 = nn.Conv2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.inception_4e_3x3 = nn.Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_1x1 = nn.Conv2d(832, 256, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_5x5_reduce = nn.Conv2d(832, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_3x3_reduce = nn.Conv2d(832, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_pool_proj = nn.Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_5x5 = nn.Conv2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.inception_5a_3x3 = nn.Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_3x3_reduce = nn.Conv2d(832, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_5x5_reduce = nn.Conv2d(832, 48, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_1x1 = nn.Conv2d(832, 384, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_pool_proj = nn.Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_3x3 = nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_5x5 = nn.Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.loss3_classifier_1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        conv1_7x7_s2     = self.conv1_relu_7x7(self.conv1_7x7_s2(x))
        pool1_3x3_s2     = self.pool1_3x3_s2(conv1_7x7_s2)
        pool1_norm1      = self.LRN(size=5, alpha=0.0001, beta=0.75)(pool1_3x3_s2)
        conv2_3x3_reduce = self.conv2_relu_3x3_reduce(self.conv2_3x3_reduce(pool1_norm1))
        conv2_3x3        = self.conv2_relu_3x3(self.conv2_3x3(conv2_3x3_reduce))
        conv2_norm2      = self.LRN(size=5, alpha=0.0001, beta=0.75)(conv2_3x3)
        pool2_3x3_s2     = self.pool2_3x3_s2(conv2_norm2)

        # Inception 3a
        inception_3a_1x1        = F.relu(self.inception_3a_1x1(pool2_3x3_s2))
        inception_3a_3x3_reduce = F.relu(self.inception_3a_3x3_reduce(pool2_3x3_s2))       
        inception_3a_3x3        = F.relu(self.inception_3a_3x3(inception_3a_3x3_reduce))
        inception_3a_5x5_reduce = F.relu(self.inception_3a_5x5_reduce(pool2_3x3_s2))
        inception_3a_5x5        = F.relu(self.inception_3a_5x5(inception_3a_5x5_reduce))
        inception_3a_pool       = self.max_pool(pool2_3x3_s2)
        inception_3a_pool_proj  = F.relu(self.inception_3a_pool_proj(inception_3a_pool))
        inception_3a_output     = torch.cat((inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj), 1)

        # Inception 3b
        inception_3b_1x1 = F.relu(self.inception_3b_1x1(inception_3a_output))
        inception_3b_3x3_reduce = F.relu(self.inception_3b_3x3_reduce(inception_3a_output))
        inception_3b_3x3 = F.relu(self.inception_3b_3x3(inception_3b_3x3_reduce))
        inception_3b_5x5_reduce = F.relu(self.inception_3b_5x5_reduce(inception_3a_output))
        inception_3b_5x5 = F.relu(self.inception_3b_5x5(inception_3b_5x5_reduce))
        inception_3b_pool = self.max_pool(inception_3a_output)
        inception_3b_pool_proj = F.relu(self.inception_3b_pool_proj(inception_3b_pool))
        inception_3b_output = torch.cat((inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj), 1)

        pool3_3x3_s2    = F.max_pool2d(inception_3b_output, kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)
        
        # Inception 4a
        inception_4a_1x1 = F.relu(self.inception_4a_1x1(pool3_3x3_s2))
        inception_4a_3x3_reduce = F.relu(self.inception_4a_3x3_reduce(pool3_3x3_s2))
        inception_4a_3x3 = F.relu(self.inception_4a_3x3(inception_4a_3x3_reduce))
        inception_4a_5x5_reduce = F.relu(self.inception_4a_5x5_reduce(pool3_3x3_s2))
        inception_4a_5x5 = F.relu(self.inception_4a_5x5(inception_4a_5x5_reduce))
        inception_4a_pool = self.max_pool(pool3_3x3_s2)
        inception_4a_pool_proj = F.relu(self.inception_4a_pool_proj(inception_4a_pool))
        inception_4a_output = torch.cat((inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj), 1)
                
        # Inception 4b
        inception_4b_1x1 = F.relu(self.inception_4b_1x1(inception_4a_output))
        inception_4b_3x3_reduce = F.relu(self.inception_4b_3x3_reduce(inception_4a_output))
        inception_4b_3x3 = F.relu(self.inception_4b_3x3(inception_4b_3x3_reduce))
        inception_4b_5x5_reduce = F.relu(self.inception_4b_5x5_reduce(inception_4a_output))
        inception_4b_5x5 = F.relu(self.inception_4b_5x5(inception_4b_5x5_reduce))
        inception_4b_pool = self.max_pool(inception_4a_output)
        inception_4b_pool_proj = F.relu(self.inception_4b_pool_proj(inception_4b_pool))
        inception_4b_output = torch.cat((inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj), 1)
        
        # Inception 4c
        inception_4c_1x1 = F.relu(self.inception_4c_1x1(inception_4b_output))
        inception_4c_3x3_reduce = F.relu(self.inception_4c_3x3_reduce(inception_4b_output))
        inception_4c_3x3 = F.relu(self.inception_4c_3x3(inception_4c_3x3_reduce))
        inception_4c_5x5_reduce = F.relu(self.inception_4c_5x5_reduce(inception_4b_output))
        inception_4c_5x5 = F.relu(self.inception_4c_5x5(inception_4c_5x5_reduce))
        inception_4c_pool = self.max_pool(inception_4b_output)
        inception_4c_pool_proj = F.relu(self.inception_4c_pool_proj(inception_4c_pool))
        inception_4c_output = torch.cat((inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj), 1)

        # Inception 4d
        inception_4d_1x1 = F.relu(self.inception_4d_1x1(inception_4c_output))
        inception_4d_3x3_reduce = F.relu(self.inception_4d_3x3_reduce(inception_4c_output))
        inception_4d_3x3 = F.relu(self.inception_4d_3x3(inception_4d_3x3_reduce))
        inception_4d_5x5_reduce = F.relu(self.inception_4d_5x5_reduce(inception_4c_output))
        inception_4d_5x5 = F.relu(self.inception_4d_5x5(inception_4d_5x5_reduce))
        inception_4d_pool = self.max_pool(inception_4c_output)
        inception_4d_pool_proj = F.relu(self.inception_4d_pool_proj(inception_4d_pool))
        inception_4d_output = torch.cat((inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj), 1)

        # Inception 4e
        inception_4e_1x1 = F.relu(self.inception_4e_1x1(inception_4d_output))
        inception_4e_3x3_reduce = F.relu(self.inception_4e_3x3_reduce(inception_4d_output))
        inception_4e_3x3 = F.relu(self.inception_4e_3x3(inception_4e_3x3_reduce))
        inception_4e_5x5_reduce = F.relu(self.inception_4e_5x5_reduce(inception_4d_output))
        inception_4e_5x5 = F.relu(self.inception_4e_5x5(inception_4e_5x5_reduce))
        inception_4e_pool = self.max_pool(inception_4d_output)
        inception_4e_pool_proj = F.relu(self.inception_4e_pool_proj(inception_4e_pool))
        inception_4e_output = torch.cat((inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj), 1)

        pool4_3x3_s2    = F.max_pool2d(inception_4e_output, kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)

        # Inception 5a
        inception_5a_1x1 = F.relu(self.inception_5a_1x1(pool4_3x3_s2))
        inception_5a_3x3_reduce = F.relu(self.inception_5a_3x3_reduce(pool4_3x3_s2))
        inception_5a_3x3 = F.relu(self.inception_5a_3x3(inception_5a_3x3_reduce))
        inception_5a_5x5_reduce = F.relu(self.inception_5a_5x5_reduce(pool4_3x3_s2))
        inception_5a_5x5 = F.relu(self.inception_5a_5x5(inception_5a_5x5_reduce))
        inception_5a_pool = self.max_pool(pool4_3x3_s2)
        inception_5a_pool_proj = F.relu(self.inception_5a_pool_proj(inception_5a_pool))
        inception_5a_output = torch.cat((inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj), 1)

        # Inception 5b
        inception_5b_1x1 = F.relu(self.inception_5b_1x1(inception_5a_output))
        inception_5b_3x3_reduce = F.relu(self.inception_5b_3x3_reduce(inception_5a_output))
        inception_5b_3x3 = F.relu(self.inception_5b_3x3(inception_5b_3x3_reduce))
        inception_5b_5x5_reduce = F.relu(self.inception_5b_5x5_reduce(inception_5a_output))
        inception_5b_5x5 = F.relu(self.inception_5b_5x5(inception_5b_5x5_reduce))
        inception_5b_pool = self.max_pool(inception_5a_output)
        inception_5b_pool_proj = F.relu(self.inception_5b_pool_proj(inception_5b_pool))
        inception_5b_output = torch.cat((inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj), 1)

        pool5_7x7_s1    = F.avg_pool2d(inception_5b_output, kernel_size=(7, 7), stride=(1, 1), padding=0)
        pool5_drop_7x7_s1 = F.dropout(input = pool5_7x7_s1, p = 0.5, training = self.training, inplace = True)
        loss3_classifier_0 = pool5_drop_7x7_s1.view(pool5_drop_7x7_s1.size(0), -1)
        loss3_classifier_1 = self.loss3_classifier_1(loss3_classifier_0)
        
        return loss3_classifier_1


    class LRN(nn.Module):
        def __init__(self, size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
            super(GoogLeNet.LRN, self).__init__()
            self.ACROSS_CHANNELS = ACROSS_CHANNELS
            if self.ACROSS_CHANNELS:
                self.average=nn.AvgPool3d(kernel_size=(size, 1, 1),
                        stride=1,
                        padding=(int((size-1.0)/2), 0, 0))
            else:
                self.average=nn.AvgPool2d(kernel_size=size,
                        stride=1,
                        padding=int((size-1.0)/2))
            self.alpha = alpha
            self.beta = beta

        def forward(self, x):
            if self.ACROSS_CHANNELS:
                div = x.pow(2).unsqueeze(1)
                div = self.average(div).squeeze(1)
                div = div.mul(self.alpha).add(1.0).pow(self.beta)
            else:
                div = x.pow(2)
                div = self.average(div)
                div = div.mul(self.alpha).add(1.0).pow(self.beta)
            x = x.div(div)
            return x


def googlenet(num_classes=1000, pretrained='imagenet'):
    r"""GoogLeNet model architecture from <https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf>`_ paper.
    """
    model = GoogLeNet(num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['googlenet'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        model.load_state_dict(torch.load(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model