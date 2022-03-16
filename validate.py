import argparse
from datasets import create_dataset
from utils import parse_configuration
from models import create_model
import os
from utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

"""Performs validation of a specified model.

Input params:
    config_file: Either a string with the path to the JSON
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and
        model-specific settings.
"""
def validate(configuration):

    print('Initializing dataset...')
    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of validation samples = {0}'.format(val_dataset_size))

    # for space_step in np.arange(0, 1.1, 0.1):
    print('Initializing model...')
    # configuration['model_params']['space_step'] = space_step
    model = create_model(configuration['model_params'])
    model.setup()
    model.eval()

    print('Initializing visualization...')
    visualizer = Visualizer(configuration['visualization_params_validation'])   # create a visualizer that displays images and plots

    model.pre_epoch_callback(configuration['model_params']['load_checkpoint'])

    #Loops through all validation data and runs though model
    for i, data in enumerate(val_dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        img = model.input[0].permute(1, 2, 0).cpu().detach().numpy()
        trg = model.target[0].permute(1, 2, 0).cpu().detach().numpy()
        out_img = model.output[0].permute(1, 2, 0).cpu().detach().numpy()
        out_trg_img = model.output_trg[0].permute(1, 2, 0).cpu().detach().numpy()

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(img)
        axs[0, 1].imshow(trg)
        axs[1, 0].imshow(out_img)
        axs[1, 1].imshow(out_trg_img)
        plt.savefig("./experiments/space_step_db/output_{}.png".format(i))
        # plt.show()


    #Where results are calculated and visualized
    # model.post_epoch_callback(configuration['model_params']['load_checkpoint'], visualizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()

    print('Reading config file...')
    configuration = parse_configuration(args.configfile)
    if (configuration['model_params']['load_checkpoint'] == -2):
        for epoch in range(configuration['model_params']['epoch_list'][0], configuration['model_params']['epoch_list'][1]):
            configuration['model_params']['load_checkpoint'] = epoch
            validate(configuration)
    else:
        validate(configuration)
