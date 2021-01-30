import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
import random

def deconv(c_in, c_out, k_size, stride = 2, pad = 1, bn = True): 
  """customize deconvolutional layer. Using block (deconv, batch norm)
     args: 
     - c_in: channels in
     - c_out: channels out
     - k_size: kernel size
     - stride: stride
     - pad : padding
     - bn : batch normalization
  """
  layers = []
  layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
  if bn:
    layers.append(nn.BatchNorm2d(c_out))
  return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride = 2, pad = 1, bn = True):
  """customize convolutional layer. Using block (conv, batch norm)
     args: 
     - c_in: channels in
     - c_out: channels out
     - k_size: kernel size
     - stride: stride
     - pad : padding
     - bn : batch normalization  
  """
  layers = []
  layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias = False))
  if bn:
    layers.append(nn.BatchNorm2d(c_out))
  return nn.Sequential(*layers)

def normal_init(m, mean = 0.0, std = 0.02):
  """ initial weight for model.
      args: 
      - m : model
      - mean: mean weight
      - std: standard deviation weight
  """
  if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
      m.weight.data.normal_(mean, std)
      if m.bias:
        m.bias.data.zero_()

class Generator(nn.Module):
  """Generator model. Using 3 main block: encoding, residual and decoding. Using tanh
      as the activation function for image output.
      args: 
      - conv_dim : number of base channels to be multiplied.
  """
  def __init__(self, conv_dim):
    super(Generator, self).__init__()
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

class Discriminator(nn.Module):
  """Discriminator model: using 1 block (conv,leaky_relu) as input 
      and 3 block (conv,spectral_norm,leaky_relu) as hidden layer. 
      Output is a matrix image. For discrminator do not using tanh or any activation function
    
    args:
      - conv_dim: number of base channels to be multiplied.
  """
  def __init__(self, conv_dim):
    super(Discriminator, self).__init__()
    self.conv1 = nn.Conv2d(1, conv_dim, 4, 2, 1, bias = False)
    self.conv2 = nn.utils.spectral_norm(nn.Conv2d(conv_dim, conv_dim*2, 4, 2, 1, bias = False))
    self.conv3 = nn.utils.spectral_norm(nn.Conv2d(conv_dim*2, conv_dim*4, 4, 2, 1, bias = False))
    self.conv4 = nn.utils.spectral_norm(nn.Conv2d(conv_dim*4, conv_dim*8, 4, 1, 0, bias = False))
    self.fc = nn.utils.spectral_norm(nn.Conv2d(conv_dim*8, 1, 4, 1, 0,bias = False))
  
  # weight_init
  def weight_init(self, mean, std):
      for m in self.children():
        m.apply(normal_init)

  def forward(self, x):
    out = F.leaky_relu(self.conv1(x), 0.05)
    out = F.leaky_relu(self.conv2(out), 0.05)
    out = F.leaky_relu(self.conv3(out), 0.05)
    out = F.leaky_relu(self.conv4(out), 0.05)
    out = self.fc(out).squeeze() 
    return out

if __name__ == "__main__":
    G12 = Generator(64)
    print(summary(G12,(1,64,64)))
    D1 = Discriminator(64)
    print(summary(D1,(1,64,64)))

