import torch.nn as nn
from einops import rearrange
from vlkit.ops.upsample import Upsample2d
from mmcv.cnn.resnet import Bottleneck, BasicBlock
import mmcv
from .buildingblocks import FourConv, create_encoders, create_decoders, number_of_features_per_level
import sys

class Block(nn.Module):
    def __init__(self, in_channels, channels) -> None:
        super().__init__()
        self.block = BasicBlock(
            in_channels,
            channels,
            stride=2,
            downsample=nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1, bias=False)
        )
    def forward(self, input):
        return self.block(input)

class Abstract3DUNet(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='icl',
                 num_groups=8, num_levels=3, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, **kwargs):
        super(Abstract3DUNet, self).__init__()
        
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        
        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        upsample=True)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        self.final_activation = nn.Sigmoid()


    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        logits = self.final_conv(x)
        pred = self.final_activation(logits)

        return logits, pred


class nnUNet3D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='icl',
                 num_groups=8, num_levels=3, is_segmentation=True, conv_padding=1, **kwargs):
        super(nnUNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=FourConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     **kwargs)


class nnUNet25D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='icl',
                 num_groups=8, num_levels=3, is_segmentation=True, conv_padding=1, **kwargs):
        super(nnUNet25D, self).__init__(in_channels=f_maps,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=FourConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     **kwargs)
        self.encode2d = nn.ModuleList([
            Block(in_channels, f_maps),
            Block(f_maps, f_maps),
        ])
        self.decode2d = nn.ModuleList([
            Upsample2d(f_maps, factor=2),
            Upsample2d(f_maps, factor=2),
        ])
    def forward(self, x):
        
        d, h, w = x.shape[-3:]

        # 3d -> 2d
        x = rearrange(x, 'n c d h w -> (n d) c h w')

        feat2d = []
        for e in self.encode2d:
            x = e(x)
            feat2d.append(x)
        feat2d.reverse()

        # 2d -> 3d
        x = rearrange(x, '(n d) c h w -> n c d h w', d=d)
        # encoder part

        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        # 3d -> 2d
        x = rearrange(x, 'n c d h w -> (n d) c h w')
        for de, f in zip(self.decode2d, feat2d):
            x = de(x + f)
        # 2d -> 3d
        x = rearrange(x, '(n d) c h w -> n c d h w', d=d)

        logits = self.final_conv(x)
        pred = self.final_activation(logits)

        return logits, pred