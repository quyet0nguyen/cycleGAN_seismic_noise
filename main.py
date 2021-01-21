import os
from argparse import ArgumentParser
import train
 
#get arguments from commandline
def get_args():
    parser = ArgumentParser(description='generate seismic image using cycleGAN')
    parser.add_argument('--num_epochs', type=int, default = 100)
    parser.add_argument('--batch_size', type=int, default = 64)
    parser.add_argument('--lr', type=float, default =0.0002)
    parser.add_argument('--image_size', type=int, default = 32)
    parser.add_argument('--beta1', type=float, default = 0.5)
    parser.add_argument('--dir', type=str, default='../AiCrowdData/data_train/data.npy')
    parser.add_argument('--num_iter_train', type=int, default = 900)
    args = parser.parse_args()
    return args

def main():
    
    args = get_args()
    train.main(args)

if __name__ == "__main__":
    main()

