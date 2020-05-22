from models import *
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from prep_data import *
from matplotlib import pyplot as plt
import torchvision

train_loader = get_data()

G = Generator(256, 1)
D = Discriminator(1)

outer_noise = torch.randn(128, 256, 1, 1)

optimizerD = optim.Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))
criterion = nn.BCELoss()

writer_real = SummaryWriter(f'runs/DCGANs/Real')
writer_fake = SummaryWriter(f'runs/DCGANs/Fake')

TRAIN = True

def plot_img(img1, img2):
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.show()

def train(G, D, optimizerG, optimizerD, criterion, train, EPOCHS):
    print('Training')

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        for i, (x, _) in enumerate(train):
            # training the Discriminator
            # actual data
            bs = x.size(0)
            D.zero_grad()
            label_real = torch.ones(bs)*0.9 # hack to get better adversarial examples
            D_out_real = D(x.float()).view(-1)
            lossD_real = criterion(D_out_real, label_real)
            D_x = D_out_real.mean().item()

            noise = torch.randn(bs, 256, 1, 1)
            # generated fake image
            fake = G(noise.float())
            # close to zero but not zero, same hack
            label_fake = torch.ones(bs) * 0.1

            D_out_fake = D(fake.detach().float()).view(-1)
            lossD_fake = criterion(D_out_fake, label_fake)

            D_loss = lossD_real + lossD_fake
            D_loss.backward()
            optimizerD.step()

            # training the Generator
            G.zero_grad()
            label = torch.ones(bs)
            # note we are not detaching here, since the output of generator
            # requires gradients
            generated_out = D(fake.float()).view(-1)
            G_loss = criterion(generated_out, label)
            G_loss.backward()
            optimizerG.step()

            if i % 100 == 99:
                print(f'Epoch {epoch+1}:[{i+1}/{len(train)}] Loss D: {D_loss.item()} Loss G: {G_loss.item()} D_x: {D_x}')

                try:
                    with torch.no_grad():
                        fake = G(outer_noise.float())

                        img_grid_real = torchvision.utils.make_grid(x[:64], normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake[:64], normalize=True)

                        writer_real.add_image('Real Image', img_grid_real)
                        writer_fake.add_image('Fake Image', img_grid_fake)
                        print('Logged')

                except Exception as e:
                    # we try to add, else we pass
                    print('Error adding grid image')
                    pass
    return G, D

if TRAIN:
    G, D = train(G, D, optimizerG, optimizerD, criterion, train_loader, 10)
    torch.save(G.state_dict(), 'models/Generator.pt')
    torch.save(D.state_dict(), 'models/Discriminator.pt')
else:
    G.load_state_dict(torch.load('./models/Generator.pt'))
    D.load_state_dict(torch.load('./models/Discriminator.pt'))
    
