import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet50(nn.Module):
    """Pretrained ResNet50 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 4

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        256: 1,  # First layer features
        512: 2,  # Second layer features
        1024: 3,  # Third layer features
        2048: 4  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=False,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained ResNet50 for SwAV_SIFID

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of first layer features
                - 2: corresponds to output of second layer features
                - 3: corresponds to output of third layer features
                - 4: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(ResNet50, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 4, \
            'Last possible output block index is 4'

        self.blocks = nn.ModuleList()

        resnet50 = models.resnet50(pretrained=False)
        # state_dict = torch.load(os.path.join(os.path.dirname(__file__), 'swav_800ep_pretrain.pth.tar'))
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # resnet50.load_state_dict(state_dict, strict=False)

        # Block 0: input to maxpool
        block0 = [
                resnet50.conv1,
                resnet50.bn1,
                resnet50.relu,
            ]
        self.blocks.append(nn.Sequential(*block0))
        state_dict = torch.load(os.path.join(os.path.dirname(__file__), 'swav_800ep_block0.pth'))
        self.blocks.load_state_dict(state_dict)
        # torch.save(self.blocks.state_dict(), os.path.join(os.path.dirname(__file__), 'swav_800ep_block0.pth'))
        # exit()

        # Block 1: layer1
        if self.last_needed_block >= 1:
            block1 = [
                nn.MaxPool2d(kernel_size=3, stride=2),
                resnet50.layer1
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: layer2
        if self.last_needed_block >= 2:
            block2 = [
                resnet50.layer2
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: layer3
        if self.last_needed_block >= 3:
            block3 = [
                resnet50.layer3
            ]
            self.blocks.append(nn.Sequential(*block3))

        # Block 4: layer4
        if self.last_needed_block >= 4:
            block4 = [
                resnet50.layer4,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block4))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.upsample(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
