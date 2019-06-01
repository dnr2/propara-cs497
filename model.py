import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
from collections import defaultdict

##################################################################################
### Utils
##################################################################################        

class NeuralModel:
    """
    Class for storing neural network TF variables.
    Also trains and evaluates the model.
    """
    
    def __init__(self, sess, loss, optimizer, x_data, y_target, prediction, saver):
        """
        Arguments:
        
        sess = tensorflow session
        loss = model's loss
        optimizer = model's optimizer
        x_data = model input (context)
        y_target = model expected output
        prediction = prediction (index for predicted word given context)
        """
        self.sess = sess
        self.loss = loss
        self.optimizer = optimizer
        self.x_data = x_data
        self.y_target = y_target
        self.prediction = prediction
        self.saver = saver
            
    def plot_cost_history(cost_history, fig_filename = None):
        plt.plot(cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Epoch')
        if fig_filename is None:
            plt.show()
        else:
            plt.savefig(fig_filename)

    def time_now_str():
        return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    ##################################################################################
    ### Model
    ##################################################################################

    def create_model(params):  
        tf.reset_default_graph()

        # declare the training data placeholders
        x_data = tf.placeholder(tf.float32, [None, None], name="input")
        y_target = tf.placeholder(tf.float32, [None, None], name="output")
        
        # TODO: declare variables and make computations
        
        # TODO: compute loss and prediction
        
        # add optimiser and checkpoint
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        optimizer = optimizer.minimize(loss)
        saver = tf.train.Saver()
        
        # initialize and create session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        # add variables to model structure
        model = NeuralModel(sess, loss, optimizer, x_data, y_target, prediction, saver)
        return model
        
    ##################################################################################
    ### Training and Testing
    ##################################################################################

    def create_batch(params, batch_idxs = None):
        # TODO: figure out how to create batch
        pass

    def train_model(model, params):
        # get arguments from params
        epochs, batch_sz = params.epochs, params.batch_sz
        toks_idxs = params.toks_idxs
        cost_history = []
        
        # execute min-batch training
        total_batch = int(len(toks_idxs) / batch_sz)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                # create batch and train on generated data.
                x_batch, y_batch = create_batch(params)
                _, cost = model.sess.run([model.optimizer, model.loss], 
                                         feed_dict={model.x_data: x_batch, model.y_target: y_batch})
                avg_cost += cost
                if i == 0 and epoch == 0:
                    # intial cost (random NN)
                    cost_history.append(avg_cost)
                    if DEBUG == True:
                        print('\nEpoch:', '%04d' % epoch, 'cost =', '{:.6f}'.format(avg_cost))
                        
            # TODO: compute validation accuracy.
            
            avg_cost /= total_batch
            cost_history.append(avg_cost)
            
            # save current state of model
            save_path = model.saver.save(model.sess, os.path.join(os.getcwd(), "checkpoints", "model.ckpt"))
            print("Model saved in path: %s" % save_path)
            
            if DEBUG == True:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(avg_cost))
                    
        return cost_history

    def evaluate_model(model, params):
        # TODO: evaluate model
        pass