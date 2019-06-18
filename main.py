# imports

import os
import datetime
import argparse

from loader import Loader
from model import NeuralModel

##################################################################################
### Constants
##################################################################################

DEBUG = True # Set to true if you want to print debugging information.

##################################################################################
### Structures
##################################################################################

# classes / structures

class Parameters:
    """Class storing all model parameters"""
    
    def __init__(self, lr, epochs, checkpoint_path):
        """
        Arguments:
        lr = learning rate
        epochs = number of epochs
        """
        self.lr = lr
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path

##################################################################################
### Main
##################################################################################

def main(args):
    # load ProPara dataset and GloVe embeddings
    loader = Loader()
    loader.load_data()
    if args.subset is not None:
        for i in range(len(loader.part)):
            loader.data[i] = loader.data[i][:args.subset]
    
    model = None
    checkpoint_path = None 
    if args.checkpoint is not None:
        checkpoint_path = os.path.join(os.getcwd(), "checkpoints", args.checkpoint)
    params = Parameters(lr = args.lr, epochs = args.epochs, checkpoint_path = checkpoint_path)

    if args.train:
        # train model
        model = NeuralModel(loader = loader, params = params)
        if checkpoint_path is not None:
            model.restore_model(checkpoint_path)
        else:
            model.create_model()
        cost_history = model.train_model()
        model.plot_and_save_cost_history(cost_history)
    if args.test:
        # test model
        if model is None:
            if checkpoint_path is None:
                raise Exception('\nIf only testing (--test), then checkpoint arg (-cp) has to be specified')
            model = NeuralModel(loader = loader, params = params)
            model.restore_model(checkpoint_path)
        model.output_test_predictions()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PROPARA - PROGLOBAL MODEL (implemented by Danilo Ribeiro and William Hancock)')
    parser.add_argument('-cp', action="store", dest="checkpoint", default=None, help="full path to saved model (checkpoint)")
    parser.add_argument('-lr', action="store", dest="lr", type=float, default=1.0, help="learning rate")
    parser.add_argument('-ep', action="store", dest="epochs", type=int, default=10, help="number of epochs")
    parser.add_argument('-ss', action="store", dest="subset", type=int, default=None, 
        help="uses subset of training and test data (i.e. from 0 to SUBSET, debug only)")
    parser.add_argument('--train', action="store_true", dest="train", help="runs training")
    parser.add_argument('--test', action="store_true", dest="test", help="runs testing")
    
    args = parser.parse_args()
    if DEBUG == True:
        print("\nARGS = ", args)
    
    main(args)