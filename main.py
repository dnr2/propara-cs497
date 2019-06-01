# imports

import os
import datetime
import argparse

from .loader import Corpus 

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
    
    def __init__(self, toks_idxs, vocab, m, h, n, lr, batch_sz, epochs):
        """
        Arguments:
        
        toks_idx = list of token indexes from corpus
        vocab = dictionary mapping from tokens (text) to indexes and vice-versa.
        """
        self.toks_idxs = toks_idxs
        self.num_toks = len(toks_idxs) # total number of tokens in corpus
        self.vocab = vocab
        self.V = int(len(vocab.keys()) / 2) # vocabulary size
        self.lr = lr
        self.batch_sz = batch_sz
        self.epochs = epochs
        # TODO: add more parameters

##################################################################################
### Main
##################################################################################

def main(args):
    # load ProPara dataset and GloVe embeddings
    loader = Loader()
    loader.load_data()
    
    model = None
    
    if args.train:
        # train model
        params = Parameters(toks_idxs = loader.idxs[TRAIN][:args.subset].copy(), vocab = loader.vocab, 
            m = args.m, h = args.h, n = args.n, lr = args.lr, batch_sz = args.bs, epochs = args.epochs)
        model = create_model(params)
        if args.checkpoint is not None:
            model.saver.restore(model.sess, args.checkpoint)
        
        cost_history = train_model(model, params)
        plot_cost_history(cost_history, fig_filename = os.path.join(os.getcwd(), "plots",
            'cost_history_' + time_now_str() + '.png'))
    if args.test:
        # test model
        params = Parameters(toks_idxs = loader.idxs[TEST][:args.subset].copy(), vocab = loader.vocab, 
            m = args.m, h = args.h, n = args.n, lr = args.lr, batch_sz = args.bs, epochs = args.epochs)
        if model is None:
            if args.checkpoint is None:
                raise Exception('\nIf only testing (--test), then checkpoint arg (-cp) has to be specified')
            model = create_model(params)
            model.saver.restore(model.sess, args.checkpoint)
        
        evaluate_model(model, params)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PROPARA - PROGLOBAL MODEL (implemented by Danilo Ribeiro and William Hancock)')
    parser.add_argument('-cp', action="store", dest="checkpoint", default=None, help="full path to saved model (checkpoint)")
    parser.add_argument('-lr', action="store", dest="lr", type=float, default=0.1, help="learning rate")
    parser.add_argument('-bs', action="store", dest="bs", type=int, default=30, help="batch size")
    parser.add_argument('-ep', action="store", dest="epochs", type=int, default=10, help="number of epochs")
    parser.add_argument('-ss', action="store", dest="subset", type=int, default=None, 
        help="uses subset of training and test data (i.e. from 0 to SUBSET, debug only)")
    parser.add_argument('--train', action="store_true", dest="train", help="runs training")
    parser.add_argument('--test', action="store_true", dest="test", help="runs testing")
    
    args = parser.parse_args()
    if DEBUG == True:
        print("\nARGS = ", args)
    
    main(args)