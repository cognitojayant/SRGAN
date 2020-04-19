from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
from model import *
from dataset import *
import torch.nn as nn
import torch
import os
import numpy as np
import sys

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
device = torch.cuda.is_available()


def train(hr_shape, channels, epoch, n_epochs, lr, b1, b2, batch_size, n_cpu, checkpoint_interval,
          sample_interval):
    # Initialize generator and discriminator
    generator = GeneratorResNet()
    discriminator = Discriminator(input_shape=(channels, *hr_shape))
    feature_extractor = FeatureExtractor()

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_gan = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()

    if device:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        feature_extractor = feature_extractor.cuda()
        criterion_gan = criterion_gan.cuda()
        criterion_content = criterion_content.cuda()

    if epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
        discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))

    # Optimizers
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_des = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    Tensor = torch.cuda.FloatTensor if device else torch.Tensor

    imageloader = ImageDataLoader(batch_size, num_worker=n_cpu)
    dataloader = imageloader.image_dataloader('images/scanned', hr_shape=hr_shape)


    # ----------
    #  Training
    # ----------

    for epoch in range(epoch, n_epochs):
        for i, imgs in enumerate(dataloader):


            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_gen.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Adversarial loss
            loss_GAN = criterion_gan(discriminator(gen_hr), valid)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())

            # Total loss
            loss_g = loss_content + 1e-3 * loss_GAN

            loss_g.backward()
            optimizer_gen.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_des.zero_grad()

            # Loss of real and fake images
            loss_real = criterion_gan(discriminator(imgs_hr), valid)
            loss_fake = criterion_gan(discriminator(gen_hr.detach()), fake)

            # Total loss
            loss_d = (loss_real + loss_fake) / 2

            loss_d.backward()
            optimizer_des.step()

            # --------------
            #  Log Progress
            # --------------

            sys.stdout.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(dataloader), loss_d.item(), loss_g.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                # Save image grid with upsampled inputs and SRGAN outputs
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                img_grid = torch.cat((imgs_lr, gen_hr), -1)
                save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
            torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
