import torch.optim as optim
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torchvision.utils as vutils 
import model
import residual_model
import data
from torch.utils.tensorboard import SummaryWriter

def train(G_12, G_21, D_1, D_2, optimizer_G, optimizer_D, batch_size, num_epochs, X_train, noised_train_data, writer):
    
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

    for epoch in range(num_epochs):

        print('Starting epoch {}...'.format(epoch), end=' ')
        
        noised_train_data_iter = iter(noised_train_data)
        X_train_iter = iter(X_train)

        for _ in tqdm(range(900)): #train in batch_size*900 img

            noise_iter = noised_train_data_iter.next()[0] 
            noise = Variable(noise_iter)

            seismic_iter = X_train_iter.next()[0] 
            seismic = Variable(seismic_iter)
            
            if torch.cuda.is_available():
                noise = noise.cuda()
                seismic = seismic.cuda()

            #======= Train D =======#

            #reset grad
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()


            # train with real images
            out1 = D_1(seismic.float())
            d_seismic_loss = torch.mean((out1-1)**2)

            out2 = D_2(noise.float())
            d_noise_loss = torch.mean((out2-1)**2)

            d_real_loss = d_seismic_loss + d_noise_loss
            d_real_loss.backward()
            optimizer_D.step()

            # train with fake images

            #reset grad
            optimizer_D.zero_grad()
            
            fake_noise = G_12(seismic.float())
            out1 = D_2(fake_noise.float())
            d2_loss = torch.mean((out1-1)**2)

            fake_seismic = G_21(noise.float())
            out2 = D_1(fake_seismic.float())
            d1_loss = torch.mean((out2-1)**2)

            d_fake_loss = d1_loss + d2_loss
            d_fake_loss.backward()
            optimizer_D.step()

            #======= Train G =======#

            #train mnist-svhn_mnist cycle
            fake_noise = G_12(seismic.float())
            out = D_2(fake_noise.float())
            reconst_seismic = G_21(fake_noise.float())
            
            g_loss = torch.mean((out-1)**2) + torch.mean((seismic - reconst_seismic)**2) # sum g_loss and reconst_lost

            g_loss.backward()
            optimizer_G.step()

            #train svhn_mnist_svhn cycle
            optimizer_G.zero_grad()
            fake_seismic = G_21(noise.float())
            out = D_1(fake_seismic)
            reconst_noise = G_12(fake_seismic.float())

            g_loss = torch.mean((out-1)**2) + torch.mean((noise - reconst_noise)**2) # sum g_loss and reconst_lost

            g_loss.backward()
            optimizer_G.step()

            print("d_real_loss:", d_real_loss.item()," d_seismic_loss:",d_seismic_loss.item()," d_noise_loss:", d_noise_loss.item()," d_fake_loss:", d_fake_loss.item()," g_loss:", g_loss.item())
            writer.add_scalar("d_real_loss/train", d_real_loss.item(), _ + epoch * 938)
            writer.add_scalar("d_seismic_loss/train", d_seismic_loss.item(), _ + epoch * 938)
            writer.add_scalar("d_noise_loss/train", d_noise_loss.item(), _ + epoch * 938)
            writer.add_scalar("d_fake_loss/train",d_fake_loss.item(), _ + epoch * 938)
            writer.add_scalar("g_loss/train", g_loss.item(), _ + epoch * 938)
            writer.add_scalar("epoch/train", epoch +1, _ + epoch * 938)

        fake_noise = G_12(fixed_seismic.float())
        fake_seismic = G_21(fixed_noised.float())

        grid = vutils.make_grid(fake_seismic, nrow=8, normalize=True)
        img_list_seismic.append(writer.add_image("generate images seismic", grid, epoch))

        grid = vutils.make_grid(fake_noise, nrow=8, normalize=True)
        img_list_noised.append(writer.add_image("generate images noise", grid, epoch))

        gen_12_name = 'generator_state_seismic_to_noise_' + str(epoch) + '.pt'
        gen_21_name = 'generator_state_noise_to_seismic_' + str(epoch) + '.pt'

        dis_1_name = 'discriminator_state_seismic_' + str(epoch) + '.pt'
        dis_2_name = 'discriminator_state_noise_' + str(epoch) + '.pt'
        if epoch % 10 == 0 :
            torch.save(G_12.state_dict(), gen_12_name)
            torch.save(G_21.state_dict(), gen_21_name)
            torch.save(D_1.state_dict(), dis_1_name)
            torch.save(D_2.state_dict(), dis_2_name)
    

def main(args):

    writer = SummaryWriter()

    ##=== run with model package ====#
    # G_12 = model.G12(args.batch_size)
    # G_12 = G_12.float()
    # G_21 = model.G21(args.batch_size)
    # G_21 = G_21.float()
    # D_1 = model.D1(args.batch_size)
    # D_1 = D_1.float()
    # D_2 = model.D2(args.batch_size)
    # D_2 = D_2.float()

    # G_12.weight_init(mean = 0.0, std = 0.02)
    # G_21.weight_init(mean = 0.0, std = 0.02)
    # D_1.weight_init(mean = 0.0, std = 0.02)
    # D_2.weight_init(mean = 0.0, std = 0.02)

    ##===run with residual model ====#
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
    
    optimizer_G = torch.optim.Adam(
        list(G_12.parameters()) + list(G_21.parameters()), lr=args.lr, betas=(args.beta1, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        list(D_1.parameters()) + list(D_2.parameters()), lr=args.lr, betas=(args.beta1, 0.999)
    )

    X_train, noised_train_data = data.train_dataset(args.dir, args.batch_size, args.image_size)

    #set label
    real_label = Variable(torch.ones(args.batch_size))
    fake_label = Variable(torch.zeros(args.batch_size))
    if torch.cuda.is_available():
        real_label = real_label.cuda()
        fake_label = fake_label.cuda()

    train(G_12, G_21, D_1, D_2, optimizer_G, optimizer_D, args.batch_size, args.num_epochs, X_train, noised_train_data, writer)
