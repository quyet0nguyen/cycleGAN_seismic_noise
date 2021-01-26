import data
import torch
import residual_model
from argparse import ArgumentParser
from torch.autograd import Variable
import PSNR

def get_args():
    parser = ArgumentParser(description='generate seismic image using cycleGAN')
    parser.add_argument('--batch_size', type=int, default = 64)
    parser.add_argument('--image_size', type=int, default = 32)
    parser.add_argument('--dir', type=str, default='../AiCrowdData/data_train/data.npy')
    #parser.add_argument('--num_iter_test', type=int, default=10)
    parser.add_argument('--G12_dir', type=str, default='./generator_state_seismic_to_noise_0.pt')
    parser.add_argument('--G21_dir', type=str, default='./generator_state_noise_to_seismic_0.pt')
    args = parser.parse_args()
    return args

def get_model(dir):
    model = residual_model.Generator(1,1)
    model = model.float()

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.load_state_dict(torch.load(dir))

    return model

def main():
    args = get_args()
    X_test, noised_test_data = data.train_dataset(args.dir, args.batch_size, args.image_size, 1)
    noised_iter = iter(noised_test_data)
    test_iter = iter(X_test)
    test = Variable(test_iter.next()[0])
    noised = Variable(noised_iter.next()[0])

    if torch.cuda.is_available():
        test = test.cuda()
        noised = noised.cuda()

    G_12 = get_model(args.G12_dir)
    G_21 = get_model(args.G21_dir)

    G_12.eval()
    G_21.eval()

    generate_seismic_image = G_21(test.float())
    generate_noise_image = G_12(noised.float())

    loss = PSNR()

    estimate_loss_generate_seismic = loss(generate_seismic_image, test)
    estimate_loss_noise_seismic = loss(generate_noise_image, noised)

    print(estimate_loss_generate_seismic)
    print(estimate_loss_noise_seismic)

if __name__ == "__main__":
    main()