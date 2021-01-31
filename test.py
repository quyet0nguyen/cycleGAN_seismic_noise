import data
import torch
import residual_model
from argparse import ArgumentParser
from torch.autograd import Variable
import PSNR
import model

def get_args():
    parser = ArgumentParser(description='generate seismic image using cycleGAN')
    parser.add_argument('--batch_size', type=int, default = 64)
    parser.add_argument('--image_size', type=int, default = 32)
    parser.add_argument('--dir', type=str, default='../AiCrowdData/data_train/data.npy')
    parser.add_argument('--state_dict', type=str, default="./state_dict_33_base.pth")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    Atest, Btest = data.train_dataset(args.dir, args.batch_size, args.image_size, 1)
    B_test_iter = iter(Btest)
    A_test_iter = iter(Atest)
    B_test = Variable(B_test_iter.next()[0])
    A_test = Variable(A_test_iter.next()[0])
    
    G_12 = model.Generator(64)
    G_21 = model.Generator(64)

    checkpoint = torch.load(args.state_dict)
    G_12.load_state_dict(checkpoint['G_12_state_dict'])
    G_21.load_state_dict(checkpoint['G_21_state_dict'])


    if torch.cuda.is_available():
        test = test.cuda()
        noised = noised.cuda()
        G_12 = G_12.cuda()
        G_21 = G_21.cuda()

    G_12.eval()
    G_21.eval()

    generate_A_image = G_21(B_test.float())
    grid = vutils.make_grid(generate_A_image, nrow=8, normalize=True)
    vutils.save_image(grid,"generate_A_image.png")

    generate_B_image = G_12(A_test.float())
    grid = vutils.make_grid(generate_A_image, nrow=8, normalize=True)
    vutils.save_image(grid,"generate_B_image.png")

    loss = PSNR.PSNR()

    estimate_loss_generate_A = loss(generate_A_image, A_test)
    estimate_loss_generate_B = loss(generate_B_image, B_test)

    print(estimate_loss_generate_A)
    print(estimate_loss_generate_B)

if __name__ == "__main__":
    main()
