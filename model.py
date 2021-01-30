import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

def deconv(c_in, c_out, k_size, stride = 2, pad = 1, bn = True): 
  """customize deconvolutional layer"""
  layers = []
  layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
  if bn:
    layers.append(nn.BatchNorm2d(c_out))
  return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride = 2, pad = 1, bn = True):
  """customize convolutional layer"""
  layers = []
  layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias = False))
  if bn:
    layers.append(nn.BatchNorm2d(c_out))
  return nn.Sequential(*layers)

def normal_init(m, mean = 0.0, std = 0.02):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if m.bias:
          m.bias.data.zero_()

class G12(nn.Module):
  def __init__(self, conv_dim):
    super(G12, self).__init__()
    #encoding blocks
    self.conv1 = conv(1, conv_dim, 4)
    self.conv2 = conv(conv_dim, conv_dim*2, 4)
    self.conv3 = conv(conv_dim*2, conv_dim*4, 4)

    #residual blocks
    self.conv4 = conv(conv_dim*4, conv_dim*4, 3, 1, 1)
    self.conv5 = conv(conv_dim*4, conv_dim*4, 3, 1, 1)

    #decoding blocks
    self.deconv1 = deconv(conv_dim*4, conv_dim*2, 4)
    self.deconv2 = deconv(conv_dim*2, conv_dim, 4)
    self.deconv3 = deconv(conv_dim, 1, 4, bn = False)
  
  # weight_init
  def weight_init(self, mean, std):
      for m in self.children():
        m = m.apply(normal_init)

  def forward(self, x):
    out = F.leaky_relu(self.conv1(x), 0.05)       
    out = F.leaky_relu(self.conv2(out), 0.05)
    out = F.leaky_relu(self.conv3(out), 0.05)

    out = F.leaky_relu(self.conv4(out), 0.05)      
    out = F.leaky_relu(self.conv5(out), 0.05)      

    out = F.leaky_relu(self.deconv1(out), 0.05)
    out = F.leaky_relu(self.deconv2(out), 0.05)
    out = F.tanh(self.deconv3(out))                 
    
    return out

class G21(nn.Module):
  def __init__(self, conv_dim):
    super(G21, self).__init__()

    #encoding blocks
    self.conv1 = conv(1, conv_dim, 4)
    self.conv2 = conv(conv_dim, conv_dim*2, 4)
    self.conv3 = conv(conv_dim*2, conv_dim*4, 4)

    #residual blocks
    self.conv4 = conv(conv_dim*4, conv_dim*4, 3, 1, 1)
    self.conv5 = conv(conv_dim*4, conv_dim*4, 3, 1, 1)

    #decoding blocks
    self.deconv1 = deconv(conv_dim*4, conv_dim*2, 4)
    self.deconv2 = deconv(conv_dim*2, conv_dim, 4)
    self.deconv3 = deconv(conv_dim, 1, 4, bn = False)
  
  # weight_init
  def weight_init(self, mean, std):
      for m in self.children():
        m.apply(normal_init)

  def forward(self, x):
    out = F.leaky_relu(self.conv1(x), 0.05)
    out = F.leaky_relu(self.conv2(out), 0.05)
    out = F.leaky_relu(self.conv3(out), 0.05)
    
    out = F.leaky_relu(self.conv4(out), 0.05)
    out = F.leaky_relu(self.conv5(out), 0.05)

    out = F.leaky_relu(self.deconv1(out), 0.05)
    out = F.leaky_relu(self.deconv2(out), 0.05)
    out = F.tanh(self.deconv3(out))
    return out

class D1(nn.Module):
  """Discriminator for mnist"""
  def __init__(self, conv_dim):
    super(D1, self).__init__()
    self.conv1 = nn.Conv2d(1, conv_dim, 4, bias = False)
    self.conv2 = nn.utils.spectral_norm(nn.Conv2d(conv_dim, conv_dim*2, 4, bias = False))
    self.conv3 = nn.utils.spectral_norm(nn.Conv2d(conv_dim*2, conv_dim*4, 4, bias = False))
    self.fc = nn.utils.spectral_norm(nn.Conv2d(conv_dim*4, 1, 4, 1, 0,bias = False))
  
  # weight_init
  def weight_init(self, mean, std):
      for m in self.children():
        m.apply(normal_init)

  def forward(self, x):
    out = F.leaky_relu(self.conv1(x), 0.05)
    out = F.leaky_relu(self.conv2(out), 0.05)
    out = F.leaky_relu(self.conv3(out), 0.05)
    out = self.fc(out).squeeze() 
    return out

class D2(nn.Module):
  """Discriminator for svhn"""
  def __init__(self, conv_dim):
    super(D2, self).__init__()
    self.conv1 = SpectralNorm(nn.Conv2d(1, conv_dim, 4, bias = False))
    self.conv2 = SpectralNorm(nn.Conv2d(conv_dim, conv_dim*2, 4, bias = False))
    self.conv3 = SpectralNorm(nn.Conv2d(conv_dim*2, conv_dim*4, 4, bias = False))
    self.fc = SpectralNorm(nn.Conv2d(conv_dim*4, 1, 4, 1, 0,bias = False))
  
  # weight_init
  def weight_init(self, mean, std):
      for m in self.children():
        m.apply(normal_init)

  def forward(self, x):
    out = F.leaky_relu(self.conv1(x), 0.05)
    out = F.leaky_relu(self.conv2(out), 0.05)
    out = F.leaky_relu(self.conv3(out), 0.05)
    out = self.fc(out).squeeze()
    return out

if __name__ == "__main__":
    batch_size = 64
    G_12 = G12(batch_size)
    G_21 = G21(batch_size)
    D_1 = D1(batch_size)
    D_2 = D2(batch_size)
    print(G_12)
    print(D_1)

