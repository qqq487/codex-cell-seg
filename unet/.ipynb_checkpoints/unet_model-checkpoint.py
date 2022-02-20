# """ Full assembly of the parts to form the complete network """
# from .cbam import ChannelGate
# from .unet_parts import *


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         #self.ca0 = ChannelGate(64)
#         self.down1 = Down(64, 128)
#         # self.ca1 = ChannelGate(128)
#         self.down2 = Down(128, 256)
#         # self.ca2 = ChannelGate(256)
#         self.down3 = Down(256, 512)
#         # self.ca3 = ChannelGate(512)
        
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         # self.ca4 = ChannelGate(1024 // factor)

#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.ca0(self.inc(x))
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

""" Full assembly of the parts to form the complete network """
from .cbam import ChannelGate, SpatialGate
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        #self.ca0 = ChannelGate(64)
        self.down1 = Down(64, 128)
        #self.ca1 = ChannelGate(128)
        self.down2 = Down(128, 256)
        #self.ca2 = ChannelGate(256)
        self.down3 = Down(256, 512)
        #self.ca3 = ChannelGate(512)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.sa4 = SpatialGate()
        self.ca4 = ChannelGate(1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        
        # x1 = self.ca0(self.inc(x))
        # x2 = self.ca1(self.down1(x1))
        # x3 = self.ca2(self.down2(x2))
        # x4 = self.ca3(self.down3(x3))
        x5 = self.sa4(self.ca4(self.down4(x4)))
        

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    

# """ Full assembly of the parts to form the complete network """

# from .unet_parts import *


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits
