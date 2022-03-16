import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from losses.EdgeLoss import EdgeLoss
from losses.GreyscaleLoss import GreyscaleLoss
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from torchvision.models import vgg19


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

#The abstract model class, uses above defined class and is used in the train script
class SRGANmodel(BaseModel):
    """basenet for fer2013"""

    def __init__(self, configuration):
        super().__init__(configuration)

        self.input_size = configuration['input_size']
        self.output_size = configuration['output_size']
        # self.lambda_adv = configuration['lambda_adv']
        # self.lambda_pixel = configuration['lambda_pixel']

        #Initialize model defined above
        self.model = GeneratorResNet().cuda()
        self.model.cuda()

        #Define loss function
        self.mse_loss = nn.MSELoss().cuda()
        self.l1_loss = nn.L1Loss().cuda()
        self.greyscale_loss = GreyscaleLoss().cuda()
        self.discriminator = Discriminator((3, 512, 512)).cuda() #Discriminator(img_size=self.output_size).cuda()
        self.feature_extractor = FeatureExtractor().cuda()

        #Define optimizer
        self.optimizer_g = torch.optim.Adam(
            self.model.parameters(),
            lr=configuration['lr'],
            betas=(configuration['momentum'], 0.999),
            weight_decay=configuration['weight_decay']
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=configuration['lr'],
            betas=(configuration['momentum'], 0.999),
            weight_decay=configuration['weight_decay']
        )

        #Need to include these arrays with the optimizers and names of loss functions and models
        #Will be used by other functions for saving/loading
        # self.optimizers = [self.optimizers[i] for i in range(4)]
        self.optimizers = [self.optimizer_g, self.optimizer_d]
        self.loss_names = ['g', 'd']
        self.network_names = ['model']

        self.loss_g = 0
        self.loss_d = 0

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []

    #Calls the models forwards function
    def forward(self):
        x = self.downsample
        self.output = self.model.forward(x)
        # print(self.input.shape)
        # print(self.output.shape)
        return self.output

    #Computes the loss with the specified name (in this case 'total')
    def compute_loss(self, epoch):

        valid = torch.ones((self.output.shape[0], *self.discriminator.output_shape), requires_grad=False).to(self.device)
        fake = torch.zeros((self.output.shape[0], *self.discriminator.output_shape), requires_grad=False).to(self.device)

        out_features = self.feature_extractor(self.output)
        in_features = self.feature_extractor(self.input)

        gen_logits = self.discriminator(self.output)
        real_logits = self.discriminator(self.input).detach()

        self.loss_g = self.l1_loss(out_features, in_features) + self.criterion_loss(gen_logits, valid) + self.greyscale_loss(self.output, self.input)# + 1e-1 * self.edge_loss(self.output, self.input)
        # if (epoch >= self.warmup_epochs):
        #     self.loss_g = self.lambda_pixel*self.loss_g + self.lambda_adv*self.adversarial_loss(gen_logits - real_logits.mean(0, keepdim=True), valid) + self.l1_loss(out_features, in_features)

        # if (epoch >= self.warmup_epochs):
        gen_logits = self.discriminator(self.output.detach())
        real_logits = self.discriminator(self.input)
        # self.loss_d = (self.adversarial_loss(real_logits - gen_logits.mean(0, keepdim=True), valid) + self.adversarial_loss(gen_logits - real_logits.mean(0, keepdim=True), fake))/2
        self.loss_d = (self.criterion_loss(real_logits, valid) + self.criterion_loss(gen_logits, fake))/2

    #Compute backpropogation for the model
    def optimize_parameters(self, epoch):
        # self.loss_total.backward()
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        self.loss_g.backward()
        self.optimizer_g.step()
        self.optimizer_g.zero_grad()
        # if (epoch > self.warmup_epochs):
        self.loss_d.backward()
        self.optimizer_d.step()
        self.optimizer_d.zero_grad()
        torch.cuda.empty_cache()

    #Test function for the model
    def test(self):
        super().test() # run the forward pass

        # save predictions and labels as flat tensors
        self.val_images.append(self.input)
        self.val_predictions.append(self.output)
        self.val_labels.append(self.input)

    #Should be run after each epoch, outputs accuracy
    def post_epoch_callback(self, epoch, visualizer):
        self.val_predictions = torch.cat(self.val_predictions, dim=0)
        predictions = torch.argmax(self.val_predictions, dim=1)
        predictions = torch.flatten(predictions).cpu()

        self.val_labels = torch.cat(self.val_labels, dim=0)
        labels = torch.flatten(self.val_labels).cpu()

        self.val_images = torch.squeeze(torch.cat(self.val_images, dim=0)).cpu()

        # Calculate and show accuracy
        val_accuracy = accuracy_score(labels, predictions)

        metrics = OrderedDict()
        metrics['Accuracy'] = val_accuracy

        if (visualizer != None):
            visualizer.plot_current_validation_metrics(epoch, metrics)
        print('Validation accuracy: {0:.3f}'.format(val_accuracy))

        # Here you may do something else with the validation data such as
        # displaying the validation images or calculating the ROC curve

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []


if __name__ == "__main__":
    net = TEMPLATEmodel().cuda()
    from torchsummary import summary

    print(summary(net, input_size=(1, 48, 48)))
