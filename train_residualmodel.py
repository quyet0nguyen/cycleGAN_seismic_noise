import torch
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import torchvision.utils as vutils 
import model
import residual_model
import data
import itertools
from torch.utils.tensorboard import SummaryWriter

def train(G_12, G_21, D_1, D_2, optimizer_G, optimizer_D_seismic, optimizer_D_noise, lr_scheduler_G, lr_scheduler_D_seismic,
            lr_scheduler_D_noise, batch_size, num_epochs, X_train, noised_train_data, writer, num_iter):
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    criterion_identity = torch.nn.L1Loss()
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()

    noised_train_data_iter = iter(noised_train_data)
    X_train_iter = iter(X_train)
    iter_per_epoch = min(len(noised_train_data_iter), len(X_train_iter))

    # fixed mnist and svhn for sampling
    fixed_noised = Variable(noised_train_data_iter.next()[0])
    fixed_seismic = Variable(X_train_iter.next()[0])

    if torch.cuda.is_available():
        fixed_noised = fixed_noised.cuda()
        fixed_seismic = fixed_seismic.cuda()

    img_list_seismic = []
    img_list_noised = []

    grid = vutils.make_grid(fixed_seismic, nrow=8, normalize=True)
    img_list_seismic.append(writer.add_image('fixed images seismic', grid, 0))

    grid = vutils.make_grid(fixed_noised, nrow=8, normalize=True)
    img_list_noised.append(writer.add_image('fixed images noise', grid, 0))

    target_real = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)

    fake_seismic_buffer = residual_model.ReplayBuffer()
    fake_noise_buffer = residual_model.ReplayBuffer()
    
    for epoch in range(num_epochs):
        
        noised_train_data_iter = iter(noised_train_data)
        X_train_iter = iter(X_train)

        for _ in tqdm(range(num_iter)): #train in batch_size*num_iter img

            noise_iter = noised_train_data_iter.next()[0] 
            noise = Variable(noise_iter)

            seismic_iter = X_train_iter.next()[0] 
            seismic = Variable(seismic_iter)
            
            if torch.cuda.is_available():
                noise = noise.cuda()
                seismic = seismic.cuda()


            #======= Train G =======#
            #reset grad
            optimizer_G.zero_grad()

            #=====Identyty loss
            
            # G_21(seismic) should equal seismic if real seismic if fed
            same_seismic = G_21(seismic.float())
            loss_identity_seismic = criterion_identity(same_seismic, seismic)*5.0

            # G_12(noise) should equal noise if real noise fed
            same_noise = G_12(noise.float())
            loss_identity_noise = criterion_identity(same_noise, noise)*5.0

            #=====Gan Loss

            #train mnist-svhn_mnist cycle
            fake_noise = G_12(seismic.float())
            pred_fake = D_2(fake_noise.float())
            g_loss_12 = criterion_GAN(pred_fake, target_fake)

            fake_seismic = G_21(noise.float())
            pred_fake = D_1(fake_seismic.float())
            g_loss_21 = criterion_GAN(pred_fake, target_real)

            #======Cycle Loss
            recovered_seismic = G_21(fake_noise.float())
            loss_cycle_121 = criterion_cycle(recovered_seismic, seismic)*10.0

            recovered_noise = G_12(fake_seismic.float())
            loss_cycle_212 = criterion_cycle(recovered_noise, noise)*10.0

            #======Total loss
            loss_G = loss_identity_noise + loss_identity_seismic + g_loss_12 + g_loss_21 + loss_cycle_121 + loss_cycle_212
            loss_G.backward()

            optimizer_G.step()    
            #======= Train D =======#

            #======= Discriminator A
            #reset grad
            optimizer_D_seismic.zero_grad()
            
            #real loss
            pred_real = D_1(seismic.float())
            loss_D_real = criterion_GAN(pred_real, target_real)

            #fake_loss
            fake_seismic = fake_seismic_buffer.push_and_pop(fake_seismic).float()
            pred_fake = D_1(fake_seismic.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            #total loss
            loss_D_1 = (loss_D_real + loss_D_fake)*0.5
            loss_D_1.backward()

            optimizer_D_seismic.step()
            
            #===========Discriminator noise
            optimizer_D_noise.zero_grad()

            #real loss
            pred_real = D_2(noise.float())
            loss_D_real = criterion_GAN(pred_real, target_real)

            #fake loss
            fake_noise = fake_noise_buffer.push_and_pop(fake_noise).float()
            pred_fake = D_2(fake_noise.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            #total_loss
            loss_D_2 = (loss_D_real+ loss_D_fake)*0.5
            loss_D_2.backward()

            optimizer_D_noise.step()
            ################################
            writer.add_scalar("loss_G/train", loss_G, _ + epoch * num_iter)
            writer.add_scalar("loss_G_identity/train", (loss_identity_seismic+loss_identity_noise), _ + epoch * num_iter)
            writer.add_scalar("loss_G_GAN/train", (g_loss_12 + g_loss_21), _ + epoch * num_iter)
            writer.add_scalar("loss_G_cycle/train",(loss_cycle_121+ loss_cycle_212), _ + epoch * num_iter)
            writer.add_scalar("loss_D/train", (loss_D_1+loss_D_2), _ + epoch * num_iter)

        fake_noise = G_12(fixed_seismic.float())
        fake_seismic = G_21(fixed_noised.float())

        grid = vutils.make_grid(fake_seismic, nrow=8, normalize=True)
        img_list_seismic.append(writer.add_image("generate images seismic", grid, epoch))

        grid = vutils.make_grid(fake_noise, nrow=8, normalize=True)
        img_list_noised.append(writer.add_image("generate images noise", grid, epoch))

        #update learning rate 
        lr_scheduler_G.step()
        lr_scheduler_D_seismic.step()
        lr_scheduler_D_noise.step()

        gen_12_name = 'generator_state_seismic_to_noise_' + str(epoch) + '.pt'
        gen_21_name = 'generator_state_noise_to_seismic_' + str(epoch) + '.pt'

        dis_1_name = 'discriminator_state_seismic_' + str(epoch) + '.pt'
        dis_2_name = 'discriminator_state_noise_' + str(epoch) + '.pt'
        
        torch.save(G_12.state_dict(), gen_12_name)
        torch.save(G_21.state_dict(), gen_21_name)
        torch.save(D_1.state_dict(), dis_1_name)
        torch.save(D_2.state_dict(), dis_2_name)


def main(args):

    writer = SummaryWriter()

    G_12 = residual_model.Generator(1,1)
    G_12 = G_12.float()
    G_21 = residual_model.Generator(1,1)
    G_21 = G_21.float()
    D_1 = residual_model.Discriminator(1)
    D_1 = D_1.float()
    D_2 = residual_model.Discriminator(1)
    D_2 = D_2.float()

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
    optimizer_D_seismic = torch.optim.Adam(D_1.parameters(), lr=args.lr, 
        betas=(args.beta1, 0.999)
    )
    optimizer_D_noise = torch.optim.Adam(D_1.parameters(), lr=args.lr, 
        betas=(args.beta1, 0.999)
    )

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=residual_model.LambdaLR(args.num_epochs,0, round(args.num_epochs/2)).step)
    lr_scheduler_D_seismic = torch.optim.lr_scheduler.LambdaLR(optimizer_D_seismic, lr_lambda=residual_model.LambdaLR(args.num_epochs,0, round(args.num_epochs/2)).step)
    lr_scheduler_D_noise = torch.optim.lr_scheduler.LambdaLR(optimizer_D_noise, lr_lambda=residual_model.LambdaLR(args.num_epochs,0, round(args.num_epochs/2)).step)

    X_train, noised_train_data = data.train_dataset(args.dir, args.batch_size, args.image_size, args.num_iter_train)

    #set label
    real_label = Variable(torch.ones(args.batch_size))
    fake_label = Variable(torch.zeros(args.batch_size))
    if torch.cuda.is_available():
        real_label = real_label.cuda()
        fake_label = fake_label.cuda()

    train(G_12, G_21, D_1, D_2, optimizer_G, optimizer_D_seismic, optimizer_D_noise, lr_scheduler_G, lr_scheduler_D_seismic, lr_scheduler_D_noise, args.batch_size, 
        args.num_epochs, X_train, noised_train_data, writer, args.num_iter_train)
