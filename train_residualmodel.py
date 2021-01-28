import torch
from math import sqrt 
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import torchvision.utils as vutils 
import residual_model
import data
import itertools
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train(G_12, G_21, D_1, D_2, optimizer_G, optimizer_D_A, optimizer_D_B, lr_scheduler_G, lr_scheduler_D_A,
            lr_scheduler_D_B, batch_size, cur_epoch,num_epochs, A_data, B_data, writer, num_iter):

    criterion_identity = torch.nn.L1Loss()
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()

    B_data_iter = iter(B_data)
    A_data_iter = iter(A_data)

    # fixed mnist and svhn for sampling
    fixed_B = Variable(B_data_iter.next()[0])
    fixed_A = Variable(A_data_iter.next()[0])


    if torch.cuda.is_available():
        fixed_B = fixed_B.cuda()
        fixed_A = fixed_A.cuda()

    grid = vutils.make_grid(fixed_A, nrow=8, normalize=True)
    writer.add_image('fixed images A', grid, 0)

    grid = vutils.make_grid(fixed_B, nrow=8, normalize=True)
    writer.add_image('fixed images B', grid, 0)

    target_real = Variable(torch.Tensor(batch_size).fill_(1.0), requires_grad=False)
    target_fake = Variable(torch.Tensor(batch_size).fill_(0.0), requires_grad=False)

    if torch.cuda.is_available():
        target_real = target_real.cuda()
        target_fake = target_fake.cuda()

    fake_A_buffer = residual_model.ReplayBuffer()
    fake_B_buffer = residual_model.ReplayBuffer()
    
    for epoch in range(cur_epoch+1, num_epochs):
        
        B_data_iter = iter(B_data)
        A_data_iter = iter(A_data)

        for _ in tqdm(range(num_iter)): #train in batch_size*num_iter img

            B_iter = B_data_iter.next()[0]
            real_B = Variable(B_iter)

            A_iter = A_data_iter.next()[0]
            real_A = Variable(A_iter)

            if torch.cuda.is_available():
                real_B = real_B.cuda()
                real_A = real_A.cuda()


            #======= Train G =======#
            #reset grad
            optimizer_G.zero_grad()

            #=====Identyty loss
            
            # G_21(A) should equal A if real A if fed
            same_A = G_21(real_A.float())
            loss_identity_A = criterion_identity(same_A, real_A)*5.0

            # G_12(B) should equal B if real B fed
            same_B = G_12(real_B.float())
            loss_identity_B = criterion_identity(same_B, real_B)*5.0

            #=====Gan Loss

            #train mnist-svhn_mnist cycle
            fake_B = G_12(real_A.float())
            pred_fake = D_2(fake_B.float())
            g_loss_12 = criterion_GAN(pred_fake.squeeze(), target_fake)

            fake_A = G_21(real_B.float())
            pred_fake = D_1(fake_A.float())
            g_loss_21 = criterion_GAN(pred_fake.squeeze(), target_real)

            #======Cycle Loss
            recovered_A = G_21(fake_B.float())
            loss_cycle_121 = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B = G_12(fake_A.float())
            loss_cycle_212 = criterion_cycle(recovered_B, real_B)*10.0

            #======Total loss
            loss_G = loss_identity_A + loss_identity_B + g_loss_12 + g_loss_21 + loss_cycle_121 + loss_cycle_212
            loss_G.backward()

            optimizer_G.step()    
            #======= Train D =======#

            #======= Discriminator A
            #reset grad
            optimizer_D_A.zero_grad()
            
            #real loss
            pred_real = D_1(real_A.float())
            loss_D_real = criterion_GAN(pred_real.squeeze(), target_real)

            #fake_loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = D_1(fake_A.detach().float())
            loss_D_fake = criterion_GAN(pred_fake.squeeze(), target_fake)

            #total loss
            loss_D_1 = (loss_D_real + loss_D_fake)*0.5
            loss_D_1.backward()

            optimizer_D_A.step()
            
            #===========Discriminator B
            optimizer_D_B.zero_grad()

            #real loss
            pred_real = D_2(real_B.float())
            loss_D_real = criterion_GAN(pred_real.squeeze(), target_real)

            #fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = D_2(fake_B.detach().float())
            loss_D_fake = criterion_GAN(pred_fake.squeeze(), target_fake)

            #total_loss
            loss_D_2 = (loss_D_real+ loss_D_fake)*0.5
            loss_D_2.backward()

            optimizer_D_B.step()

            ################################
            writer.add_scalar("loss_G/train", loss_G, _ + epoch * num_iter)
            writer.add_scalar("loss_G_identity/train", (loss_identity_A+loss_identity_B), _ + epoch * num_iter)
            writer.add_scalar("loss_G_GAN/train", (g_loss_12 + g_loss_21), _ + epoch * num_iter)
            writer.add_scalar("loss_G_cycle/train",(loss_cycle_121+ loss_cycle_212), _ + epoch * num_iter)
            writer.add_scalar("loss_D/train", (loss_D_1+loss_D_2), _ + epoch * num_iter)

        fake_B = G_12(fixed_A.float())
        fake_A = G_21(fixed_B.float())

        grid = vutils.make_grid(fake_A, nrow=8, normalize=True)
        writer.add_image("generate images A", grid, epoch)

        grid = vutils.make_grid(fake_B, nrow=8, normalize=True)
        writer.add_image("generate images B", grid, epoch)

        #update learning rate 
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if epoch % 10 == 0:

            name = 'state_dict_' + str(epoch) + '_.pth'

            torch.save({"G_12_state_dict": G_12.state_dict(),
                    "G_21_state_dict": G_21.state_dict(),
                    "D_1_state_dict": D_1.state_dict(),
                    "D_2_state_dict": D_2.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizer_D_A" : optimizer_D_A.state_dict(),
                    "optimizer_D_B" : optimizer_D_B.state_dict(),
                    "lr_scheduler_G" : lr_scheduler_G.state_dict(),
                    "lr_scheduler_D_A" : lr_scheduler_D_A.state_dict(),
                    "lr_scheduler_D_B" : lr_scheduler_D_B.state_dict(),
                    "epoch" : epoch,
            },name)


def main(args):

    writer = SummaryWriter(log_dir='/content/cycleGAN_seismic_noise/runs/'+ datetime.now().strftime('%b%d_%H-%M-%S'))

    G_12 = residual_model.Generator(1,1)
    G_21 = residual_model.Generator(1,1)
    D_1 = residual_model.Discriminator(1)
    D_2 = residual_model.Discriminator(1)

    G_12.apply(residual_model.weights_init_normal)
    G_21.apply(residual_model.weights_init_normal)
    D_1.apply(residual_model.weights_init_normal)
    D_2.apply(residual_model.weights_init_normal)

    if torch.cuda.is_available():
        G_12 = G_12.cuda()
        G_21 = G_21.cuda()
        D_1 = D_1.cuda()
        D_2 = D_2.cuda()
    
    #=== optimizer & learning rate schedulers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_12.parameters(), G_21.parameters()), lr=args.lr, 
        betas=(args.beta1, 0.999)
    )
    optimizer_D_A = torch.optim.Adam(D_1.parameters(), lr=args.lr, 
        betas=(args.beta1, 0.999)
    )
    optimizer_D_B = torch.optim.Adam(D_1.parameters(), lr=args.lr, 
        betas=(args.beta1, 0.999)
    )

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=residual_model.LambdaLR(args.num_epochs,0, round(args.num_epochs/2)).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=residual_model.LambdaLR(args.num_epochs,0, round(args.num_epochs/2)).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=residual_model.LambdaLR(args.num_epochs,0, round(args.num_epochs/2)).step)

    # load seismic dataset
    A_data, B_data = data.train_dataset(args.dir, args.batch_size, args.image_size, args.num_iter_train)
    # load apple2orange
    # A_data = crack_dataset.dataloader(args,"train",'crack')
    # B_data = crack_dataset.dataloader(args,"train",'origin')

    model_state = args.state_dict

    if model_state!= "":
        checkpoint = torch.load(model_state)
        G_12.load_state_dict(checkpoint['G_12_state_dict'])
        G_21.load_state_dict(checkpoint['G_21_state_dict'])
        D_1.load_state_dict(checkpoint['D_1_state_dict'])
        D_2.load_state_dict(checkpoint['D_2_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
        optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
        lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
        lr_scheduler_D_A.load_state_dict(checkpoint['lr_scheduler_D_A'])
        lr_scheduler_D_B.load_state_dict(checkpoint['lr_scheduler_D_B'])
        cur_epoch = checkpoint['epoch']
    else:
        cur_epoch = 0

    train(G_12, G_21, D_1, D_2, optimizer_G, optimizer_D_A, optimizer_D_B, lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B, args.batch_size, 
        cur_epoch, args.num_epochs, A_data, B_data, writer, args.num_iter_train)
