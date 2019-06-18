import os
import datetime
import time
import sys
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, concatenate
from keras.layers import GlobalMaxPooling1D
from keras.models import Model, load_model
from keras.initializers import Constant
from keras.optimizers import Adadelta
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

"""
PROBLEMS:
1 - how to use multiple embeddings (DONE)
2 - how to concatenate each of LSTM's output (DONE)
3 - how max pooling works in this paper? (DONE)
4 - how to feed output of one part of the system to the following time step? (DONE)
5 - how big should the hidden layers be? (?)
6 - how to generate the state before first sentence (all outputs assume previous sentence)? 
"""

class NeuralModel:
    """
    Class for storing neural network TF variables.
    Also trains and evaluates the model.
    """
    
    # Constants
    MAX_SEQUENCE_LENGTH = 180
    MAX_NUM_WORDS = 2800
    GLOVE_EMBEDDING_DIM = 100
    LSTM_UNITS = 64
    NUM_CATEGORIES = 3

    def __init__(self, loader, params, verbose = True):
        """
        Class attributes:
            loader: loader containing dataset and embeddings
            verbose: if True prints debug level information.
            lstm_units: dimensionality of the LSTM output space.
        """
        self.loader = loader
        self.verbose = verbose
        self.params = params

        self.sentences = None
        self.word_embedding_matrix = None
        # define multiple input and outputs
        self.X1 = [[], [], []]
        self.X2 = [[], [], []]
        self.X3 = [[], [], []]
        self.Y1 = [[], [], []]
        # TODO: running setup for each model, even though the input, output and embedding 
        #       matrix should be the same across models.
        self.setup_input()

        self.checkpoint_path = os.path.join(
            os.getcwd(), "checkpoints", "model%s.h5" % self.time_now_str())

    ##################################################################################
    ### Utils
    ##################################################################################

    def print_progress_bar(self, value, endvalue, comment, bar_length=20):
        percent = float(value) / endvalue
        arrow = '=' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}% # {2}".format(
            arrow + spaces, int(round(percent * 100)), comment))
        sys.stdout.flush()

    def time_now_str(self):
        return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def plot_and_save_cost_history(self, cost_history, fig_filename = None):
        if fig_filename is None:
            fig_filename = os.path.join(
                os.getcwd(), "plots", 'cost_history_' + self.time_now_str() + '.png')
        plt.plot(cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Epoch')
        if fig_filename is None:
            plt.show()
        else:
            plt.savefig(fig_filename)

    def restore_model(self, file_path):
        self.checkpoint_path = file_path
        self.model = load_model(self.checkpoint_path)
        print("Model loaded from: ", self.checkpoint_path)
        print("self.model.summary()", self.model.summary())

    ##################################################################################
    ### Setup and pre-processing
    ##################################################################################

    def setup_input(self):
        self.text_preprocessing()
        self.create_word_embedding_matrix()
        self.create_input_output()

    def text_preprocessing(self):
        paragraphs = self.loader.data[self.loader.part["train"]]
        self.sentences = [sent for parag in paragraphs for sent in parag["sentences"]]
        self.tokenizer = Tokenizer(num_words=self.MAX_NUM_WORDS)
        self.tokenizer.fit_on_texts(self.sentences)
        self.word_index = self.tokenizer.word_index
        if self.verbose:
            print('num unique tokens :', len(self.word_index))

    def create_word_embedding_matrix(self):
        # prepare embedding matrix
        self.word_embedding_matrix = np.zeros((self.MAX_NUM_WORDS, self.GLOVE_EMBEDDING_DIM))
        for word, i in self.word_index.items():
            if i > self.MAX_NUM_WORDS:
                continue
            embedding_vector = self.loader.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.word_embedding_matrix[i] = embedding_vector
        
        if self.verbose:                
            print("embedding shape: ", self.word_embedding_matrix.shape)

    def _get_participant_positions(self, participant, paragraph_tokens):
        participant = participant.split(";")
        participant_tokens = [text_to_word_sequence(p)
                for p in participant]
        participant_indices = []
        for p in participant_tokens:
            p_size = len(p)
            for i in range(len(paragraph_tokens) - p_size + 1):
                if paragraph_tokens[i: i + p_size] == p:
                    participant_indices += [(i, i + p_size - 1)]
        participant_indices = list(set(participant_indices))
        participant_indices.sort()
        if len(participant_indices) == 0:
            # handle cases where no participant is found.
            participant_indices = [(0, 0)]
        return participant_indices

    def _create_participant_pos_input(self, participant_indices, sentence_offset, 
            sentence_length, paragraph_len):
        """
        Creates participant relative distance input.
        """
        y1, y2 = sentence_offset, sentence_length
        # find closest participant index according to sentence offset
        best_pos = min(participant_indices, 
            key=lambda x: 0 if x[0] >= y1 and x[1] <= y2 else 
            min(abs(y1 - x[1]), abs(x[0] - y2)))
        # calculate the distance from each token to participant
        position_input = []
        for pos in range(paragraph_len):
            if pos >= best_pos[0] and pos <= best_pos[1]:
                position_input.append(0)
            elif pos < best_pos[0]:
                position_input.append(pos - best_pos[0])
            else:
                position_input.append(pos - best_pos[1])
        return position_input

    def _create_sentece_input(self, sentence_offset, sentence_len, paragraph_len):
        """
        Creates sentence input. Feature indicating if current token is part of previous, current 
        or following sentences.
        """
        previous = [0 for i in range(0, sentence_offset)]
        current = [1 for i in range(sentence_offset, sentence_offset + sentence_len)]
        following = [2 for i in range(sentence_offset + sentence_len, paragraph_len)]
        sentence_input = previous + current + following
        return sentence_input

    def create_input_output(self):
        """
        Creates input and output for the model.
        """
        for p_name, p_idx in self.loader.part.items():
            data = self.loader.data[p_idx]
            for paragraph in data:
                paragraph_text = " ".join(paragraph["sentences"])
                paragraph_tokens = text_to_word_sequence(paragraph_text)
                paragraph_sequences = self.tokenizer.texts_to_sequences([paragraph_text])[0]
                sentence_offset = 0
                for p, participant in enumerate(paragraph["participants"]):
                    participant_indices = self._get_participant_positions(participant, paragraph_tokens)
                    for s, sentence in enumerate(paragraph["sentences"]):
                        sentence_tokens = text_to_word_sequence(sentence)
                        sentence_length = len(sentence_tokens)
                        paragraph_len = len(paragraph_tokens)
                        sentence_input = self._create_sentece_input(
                            sentence_offset, sentence_length, paragraph_len)
                        position_input = self._create_participant_pos_input(
                            participant_indices, sentence_offset, sentence_length, paragraph_len)
                        sentence_offset += len(sentence_tokens)

                        # add corresponding input.
                        # TODO: normalize input (make it between -1 and 1)
                        self.X1[p_idx].append(paragraph_sequences)
                        self.X2[p_idx].append(sentence_input)
                        self.X3[p_idx].append(position_input)

                        # Create a expected output for 3 types of states: does not exist, 
                        # location unknown, location known.
                        key = (s,p)
                        prediction = paragraph["predictions"][key]
                        self.Y1[p_idx].append(
                            [1 if prediction == i else 0 for i in range(self.NUM_CATEGORIES)])

            # pad sequences
            self.X1[p_idx] = pad_sequences(self.X1[p_idx], maxlen=self.MAX_SEQUENCE_LENGTH)
            self.X2[p_idx] = pad_sequences(self.X2[p_idx], maxlen=self.MAX_SEQUENCE_LENGTH)
            self.X3[p_idx] = pad_sequences(self.X3[p_idx], maxlen=self.MAX_SEQUENCE_LENGTH)
            self.Y1[p_idx] = np.array(self.Y1[p_idx])

    ##################################################################################
    ### Model
    ##################################################################################

    def create_model(self):
        # specifying input for sample
        input_tokens = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        input_distances = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        input_sent_pos = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        prev_cat_probs = Input(shape=(self.NUM_CATEGORIES,))        

        # create embedding layers
        tokens_emb = Embedding(output_dim=self.GLOVE_EMBEDDING_DIM, 
            input_dim=self.MAX_NUM_WORDS,
            input_length=self.MAX_SEQUENCE_LENGTH, 
            embeddings_initializer=Constant(self.word_embedding_matrix))(input_tokens)
        distances_emb = Embedding(output_dim=50, 
            input_dim=self.MAX_SEQUENCE_LENGTH,
            input_length=self.MAX_SEQUENCE_LENGTH)(input_distances)
        sent_pos_emb = Embedding(output_dim=50, 
            input_dim=4, input_length=self.MAX_SEQUENCE_LENGTH)(input_sent_pos)

        # concatenate all embedding layers 
        sequence = concatenate([tokens_emb, distances_emb, sent_pos_emb])
        lstm_encoder = Bidirectional(LSTM(self.LSTM_UNITS, return_sequences=True))(sequence)
        max_pooling = GlobalMaxPooling1D(data_format="channels_first")(lstm_encoder)
        hidden_category = concatenate([prev_cat_probs, max_pooling])
        category_output = Dense(self.NUM_CATEGORIES, activation='softmax')(hidden_category)

        # This creates a model that includes
        # the Input layer and three Dense layers
        self.model = Model(inputs=[input_tokens, input_distances, input_sent_pos, prev_cat_probs], 
            outputs=[category_output])
        self.model.compile(optimizer=Adadelta(lr=self.params.lr),
            loss='categorical_crossentropy', metrics=['accuracy'])
        print("self.model.summary()", self.model.summary())

        
    ##################################################################################
    ### Training and Testing
    ##################################################################################

    def get_inputs_and_outputs(self, part_index, sample_idx):
        x1 = self.X1[part_index][sample_idx:sample_idx+1]
        x2 = self.X2[part_index][sample_idx:sample_idx+1]
        x3 = self.X3[part_index][sample_idx:sample_idx+1]
        y1 = self.Y1[part_index][sample_idx:sample_idx+1]
        return [x1, x2, x3], [y1]

    def train_model(self):
        # get arguments from params
        train_idx = self.loader.part["train"]
        print("self.X1[train_idx].shape", self.X1[train_idx].shape)
        print("self.X2[train_idx].shape", self.X2[train_idx].shape)
        print("self.X3[train_idx].shape", self.X3[train_idx].shape)
        print("self.Y1[train_idx].shape\n", self.Y1[train_idx].shape)

        num_samples = len(self.X1[train_idx])
        cost_history = []
        cost = 0

        print("\n### Training starting")
        for epoch in range(self.params.epochs):
            start_time = time.time()
            cost = 0
            y1_prev = np.array([self.NUM_CATEGORIES * [1.0 / self.NUM_CATEGORIES]])
            print("\nEpoch %d/%d" % (epoch + 1, self.params.epochs))
            for sample_idx in range(num_samples):
                # run training and prediction on sample
                inputs, outputs = self.get_inputs_and_outputs(train_idx, sample_idx)
                inputs += [y1_prev]
                loss = self.model.train_on_batch(x=inputs, y=outputs)
                y1_prev = self.model.predict_on_batch(x=inputs)
                cost += loss[0]

                # print current stats
                if sample_idx % 10 == 0  or sample_idx == (num_samples-1):
                    avg_cost = float(cost / (sample_idx + 1))
                    elapsed_time = (time.time() - start_time) / 60.0
                    self.print_progress_bar(sample_idx + 1, num_samples, 
                        "cost: %.3f, elapsed time (min): %.2f" % (avg_cost, elapsed_time))

                # save model
                if sample_idx % 100 == 0  or sample_idx == (num_samples-1):
                    self.model.save(self.checkpoint_path)
            cost_history.append(cost/num_samples)
        self.model.save(self.checkpoint_path)
        print("\n### Training finished")
        print("final cost: ", cost/num_samples)
        print("Model saved: ", self.checkpoint_path)
        return cost_history

    def output_test_predictions(self):
        test_idx = self.loader.part["train"]
        num_samples = len(self.X1[test_idx])
        cost = 0
        start_time = time.time()
        y1_prev = np.array([self.NUM_CATEGORIES * [1.0 / self.NUM_CATEGORIES]])

        print("\n### Test starting")
        for sample_idx in range(num_samples):
            # runt prediction on test sample
            inputs, outputs = self.get_inputs_and_outputs(test_idx, sample_idx)
            inputs += [y1_prev]
            loss = self.model.test_on_batch(x=inputs, y=outputs)
            y1_prev = self.model.predict_on_batch(x=inputs)
            cost += loss[0]

            # print current stats
            if sample_idx % 10 == 0 or sample_idx == (num_samples-1):
                avg_cost = float(cost / (sample_idx + 1))
                elapsed_time = (time.time() - start_time) / 60.0
                self.print_progress_bar(sample_idx + 1, num_samples, 
                    "cost: %.3f, elapsed time (min): %.2f" % (avg_cost, elapsed_time))

        print("\n### Test finished")
        print("final cost: ", cost/num_samples)