import torch
import cv2
import numpy as np

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range (min, max)"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(y_true, y_pred):
      result = np.zeros((y_true.shape[0]))
      for i in range(y_true.shape[0]):
        max_val = max(torch.max(y_true[i,:,:,:]), torch.max(y_pred[i,:,:,:]))
        mse = torch.mean((y_true[i,:,:,:] - y_pred[i,:,:,:]) ** 2)
        result[i] = 10 * torch.log10(float(max_val) / torch.sqrt(mse))
      return result

if __name__ == '__main__':
    metric = PSNR()
    print(metric(torch.tensor([[2.,3.,5.,7.],[2.,5.,4.,1.]]), torch.tensor([[2.,3.,5.,1.],[2., 7., 3., 4.]])))
    #print(cv2.PSNR(np.array([[2.,3.,5.,7.],[2.,5.,4.,1.]]),np.array([[2.,3.,5.,1.],[2., 7., 3., 4.]])))
