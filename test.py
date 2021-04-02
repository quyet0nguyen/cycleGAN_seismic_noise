import data
import torch
import residual_model
from argparse import ArgumentParser
from torch.autograd import Variable
import torchvision.utils as vutils 
import PSNR
import model

def get_args():
    parser = ArgumentParser(description='generate seismic image using cycleGAN')
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--image_size', type=int, default = 64)
    parser.add_argument('--dir', type=str, default='../AiCrowdData/data_train/data.npy')
    parser.add_argument('--state_dict', type=str, default="./state_dict_33.pth")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    Atest, Btest = data.train_dataset(args.dir, args.batch_size, args.image_size, 1000)
    B_test_iter = iter(Btest)
    A_test_iter = iter(Atest)
    
    G_21 = model.Generator(args.batch_size)
    #G_21 = residual_model.Generator(1,1)

    checkpoint = torch.load(args.state_dict)
    G_21.load_state_dict(checkpoint['G_21_state_dict'])

    estimate_loss_generate = 0

    for i in range(1000):
      B_test = Variable(B_test_iter.next()[0])
      A_test = Variable(A_test_iter.next()[0])
      grid = vutils.make_grid(B_test, nrow=8)
      vutils.save_image(grid,"B_image.png")

      if torch.cuda.is_available():
        B_test = B_test.cuda()
        A_test = A_test.cuda()
        G_21 = G_21.cuda()

      G_21.eval()

      generate_A_image = G_21(B_test.float())
      grid = vutils.make_grid(generate_A_image, nrow=8)
      vutils.save_image(grid,"generate_A_image.png")

      loss = PSNR.PSNR()

      estimate_loss_generate = estimate_loss_generate +loss(generate_A_image, A_test)
    
    estimate_loss_generate = estimate_loss_generate /1000

    print(estimate_loss_generate)

      

if __name__ == "__main__":
    main()
