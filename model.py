import os
import datetime
import time
import sys
import numpy as np
import tensorflow as tf
import random
from keras import backend as K 
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, concatenate
from keras.layers import GlobalMaxPooling1D, multiply, Reshape, Lambda, Dropout
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.initializers import Constant
from keras.optimizers import Adadelta
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.callbacks import TensorBoard

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
    MAX_SEQUENCE_LENGTH = 150
    MAX_NUM_WORDS = 2800
    GLOVE_EMBEDDING_DIM = 50
    SENT_EMBEDDING_DIM = 5
    LSTM_UNITS_1 = 100
    LSTM_UNITS_2 = 10
    NUM_CATEGORIES = 3
    DROPOUT = 0.2

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
        self.X1 = [[], [], []] # paragraph token indexes
        self.X2 = [[], [], []] # relative distance to the participant
        self.X3 = [[], [], []] # sentence indicator (previous, current, following)
        self.Y1 = [[], [], []] # output state category 
        self.Y2 = [[], [], []] # location start position
        self.Y3 = [[], [], []] # location end position
        # map from sample index to key =  {pararaph_id, participant_id, sentence_id}
        self.sample_idx_map = [{}, {}, {}]
        # map from paragraph indexes to the list of tokens in the paragraph                                        
        self.ph_idx_tokens_map = [{}, {}, {}]
        # TODO: running setup for each model, even though the input, output and embedding 
        #       matrix should be the same across models.
        self.setup_input()

    ##################################################################################
    ### Utils
    ##################################################################################

    def print_progress_bar(self, value, endvalue, comment, bar_length=20):
        percent = float(value) / endvalue
        arrow = '=' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1:3d}% # {2}".format(
            arrow + spaces, int(round(percent * 100)), comment))
        sys.stdout.flush()

    def restore_model(self):
        self.create_model()
        self.model.load_weights(self.params.checkpoint_path)
        print("Model loaded from: ", self.params.checkpoint_path)
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
        self.tokenizer = Tokenizer(num_words=self.MAX_NUM_WORDS, filters = [])
        self.tokenizer.fit_on_texts(self.sentences)
        self.word_index = self.tokenizer.word_index
        self.reverse_word_index = {i : word for word, i in self.word_index.items()}

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

    def _tokens_similar(self, tokens_1, tokens_2):
        for t1, t2 in zip(tokens_1, tokens_2):
            if not (t1 in t2 or t2 in t1) or abs(len(t1) - len(t2)) > 2:
                return False
        return True

    def _text_to_word_sequence(self, text):
        return text_to_word_sequence(text, filters='')

    def _get_participant_positions(self, participant, paragraph_tokens):
        participant = participant.split(";")
        participant_tokens = [self._text_to_word_sequence(p)
                for p in participant]
        participant_indices = []
        for p in participant_tokens:
            p_size = len(p)
            for i in range(len(paragraph_tokens) - p_size + 1):
                if self._tokens_similar(paragraph_tokens[i: i + p_size], p):
                    participant_indices += [(i, i + p_size - 1)]
        participant_indices = list(set(participant_indices))
        participant_indices.sort()
        if len(participant_indices) == 0:
            # handle cases where no participant is found.
            participant_indices = [(0, 0)]
        return participant_indices

    def _get_before_after_locations(self, state, paragraph_tokens, sentence_offset, 
            sentence_length):
        if state == "-":
            return (-1, -1)
        if state == "?":
            return (-2, -2)
        state_tokes = self._text_to_word_sequence(state)
        state_indeces = []
        s_size = len(state_tokes)
        for i in range(len(paragraph_tokens) - s_size + 1):
            if self._tokens_similar(paragraph_tokens[i: i + s_size], state_tokes):
                state_indeces += [i]
        if len(state_indeces) == 0:
            # couldn't find location inside paragraph
            return (-2, -2)
        start_loc = min(state_indeces, 
            key = lambda x: abs(sentence_offset + int(sentence_length / 2.0) - x))
        return (start_loc, start_loc + s_size -1)

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
            sample_idx = 0
            for ph_idx, paragraph in enumerate(data):
                paragraph_text = " ".join(paragraph["sentences"])
                paragraph_tokens = self._text_to_word_sequence(paragraph_text)
                self.ph_idx_tokens_map[p_idx][ph_idx] = paragraph_tokens
                paragraph_sequences = self.tokenizer.texts_to_sequences([paragraph_text])[0]
                sentence_offset = 0
                for p, participant in enumerate(paragraph["participants"]):
                    participant_indices = self._get_participant_positions(
                        participant, paragraph_tokens)
                    for s, sentence in enumerate(paragraph["sentences"]):
                        sentence_tokens = self._text_to_word_sequence(sentence)
                        sentence_length = len(sentence_tokens)
                        paragraph_len = len(paragraph_tokens)
                        state = paragraph["states"][s+1][p]
                        sentence_input = self._create_sentece_input(
                            sentence_offset, sentence_length, paragraph_len)
                        position_input = self._create_participant_pos_input(
                            participant_indices, sentence_offset, sentence_length, paragraph_len)
                        sentence_offset += len(sentence_tokens)
                        location_start, location_end = self._get_before_after_locations(
                            state, paragraph_tokens, sentence_offset, sentence_length)

                        # add corresponding input.
                        self.X1[p_idx].append(paragraph_sequences)
                        self.X2[p_idx].append(sentence_input)
                        self.X3[p_idx].append(position_input)

                        # print("paragraph_text", paragraph_text)
                        # print("paragraph_sequences", paragraph_sequences)
                        # print(list([self.reverse_word_index[p] for p in paragraph_sequences]))
                        # print("sentence_input", sentence_input)
                        # print("position_input", position_input)
                        # input()

                        # Create a expected output for 3 types of states: does not exist, 
                        # location unknown, location known.
                        # Also adds corresponding location start and end position
                        key = (s,p)
                        prediction = paragraph["predictions"][key]
                        self.Y1[p_idx].append(
                            [1 if prediction == i else 0 for i in range(self.NUM_CATEGORIES)])
                        self.Y2[p_idx].append(
                            [1 if location_start == i else 0 
                                for i in range(self.MAX_SEQUENCE_LENGTH)])
                        self.Y3[p_idx].append(
                            [1 if location_end == i else 0 
                                for i in range(self.MAX_SEQUENCE_LENGTH)])

                        # update sample index
                        self.sample_idx_map[p_idx][sample_idx] = (ph_idx, p, s)
                        sample_idx += 1

            # pad sequences
            self.X1[p_idx] = pad_sequences(self.X1[p_idx], maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')
            self.X2[p_idx] = pad_sequences(self.X2[p_idx], maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')
            self.X3[p_idx] = pad_sequences(self.X3[p_idx], maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')            
            self.Y1[p_idx] = np.array(self.Y1[p_idx])
            self.Y3[p_idx] = np.array(self.Y2[p_idx])
            self.Y2[p_idx] = np.array(self.Y3[p_idx])

    ##################################################################################
    ### Model
    ##################################################################################

    def _location_span_module(self, prev_probs, tokens_lstm_encoder, tokens_lstm_encoder_last):
        prev_probs_reshaped = Reshape((self.MAX_SEQUENCE_LENGTH, 1))(prev_probs)
        weighted_tokens = multiply([prev_probs_reshaped, tokens_lstm_encoder])
        weighted_tokens_seq = concatenate([tokens_lstm_encoder, weighted_tokens])
        # weighted_tokens_seq = Dropout(self.DROPOUT)(weighted_tokens_seq)
        vector_rep_encoder = LSTM(self.LSTM_UNITS_2, return_sequences=False)(weighted_tokens_seq)
        vector_rep_seq = concatenate([vector_rep_encoder, tokens_lstm_encoder_last])
        # vector_rep_seq = Dropout(self.DROPOUT)(vector_rep_seq)
        output = Dense(self.MAX_SEQUENCE_LENGTH, activation='softmax')(vector_rep_seq)
        return output

    @staticmethod
    def custom_loss(idx, loss_weights):
        flat_loss_weights = K.reshape(loss_weights, (3,))
        return lambda y_true, y_pred: \
            K.gather(flat_loss_weights, idx) * categorical_crossentropy(y_true, y_pred)
        

    def write_log(self, callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()

    def create_model(self):
        # specifying input for sample
        input_tokens = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        input_distances = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        input_sent_pos = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        loss_weights = Input(shape=(3,))
        prev_cat_probs = Input(shape=(self.NUM_CATEGORIES,))
        prev_start_probs = Input(shape=(self.MAX_SEQUENCE_LENGTH,))

        # create embedding layers
        tokens_emb = Embedding(output_dim=self.GLOVE_EMBEDDING_DIM, 
            input_dim=self.MAX_NUM_WORDS,
            input_length=self.MAX_SEQUENCE_LENGTH, 
            # uses constant GloVe embeddings
            embeddings_initializer=Constant(self.word_embedding_matrix))(input_tokens)
        distances_emb = Embedding(output_dim=self.SENT_EMBEDDING_DIM, 
            input_dim=2 * self.MAX_SEQUENCE_LENGTH + 1,
            input_length=self.MAX_SEQUENCE_LENGTH)(input_distances)
        sent_pos_emb = Embedding(output_dim=self.SENT_EMBEDDING_DIM, 
            input_dim=4, input_length=self.MAX_SEQUENCE_LENGTH)(input_sent_pos)

        # concatenate all embedding layers 
        sequence = concatenate([tokens_emb, distances_emb, sent_pos_emb])
        tokens_lstm_encoder = Bidirectional(LSTM(self.LSTM_UNITS_1, return_sequences=True))(sequence)
        tokens_lstm_encoder_last = Lambda(lambda x: x[:,-1:,:])(tokens_lstm_encoder)
        tokens_lstm_encoder_last = Reshape((2 * self.LSTM_UNITS_1,))(tokens_lstm_encoder_last)

        # compute category output
        max_pooling = GlobalMaxPooling1D(data_format="channels_first")(tokens_lstm_encoder)
        hidden_category = concatenate([prev_cat_probs, max_pooling])
        # hidden_category = Dropout(self.DROPOUT)(hidden_category)
        category_output = Dense(self.NUM_CATEGORIES, activation='softmax')(hidden_category) 
        
        # compute the start and end position output
        start_output = self._location_span_module(
            prev_start_probs, tokens_lstm_encoder, tokens_lstm_encoder_last)
        end_output = self._location_span_module(
            start_output, tokens_lstm_encoder, tokens_lstm_encoder_last)
        
        # Create and compile model
        self.model = Model(inputs=[input_tokens, input_distances, input_sent_pos, 
            loss_weights, prev_cat_probs, prev_start_probs], 
            outputs=[category_output, start_output, end_output])
        self.model.compile(optimizer="adam", metrics=['acc'], # optimizer=Adadelta(lr=self.params.lr),
            loss=[self.custom_loss(idx, loss_weights) for idx in range(3)])

        if self.verbose:
            log_path = os.path.join(os.getcwd(), "logs")
            self.callback = TensorBoard(log_path)
            self.callback.set_model(self.model)
            print("self.model.summary()", self.model.summary())

        
    ##################################################################################
    ### Training and Testing
    ##################################################################################

    def _get_inputs_and_outputs(self, part_index, sample_idx, y1_prev, y2_prev):
        ph_idx, p, s = self.sample_idx_map[part_index][sample_idx]
        if (p == 0 and s == 0) or y1_prev is None:
            y1_prev = np.array([self.NUM_CATEGORIES * [1.0 / self.NUM_CATEGORIES]])
        if (p == 0 and s == 0) or y2_prev is None:
            y2_prev = np.array([self.MAX_SEQUENCE_LENGTH * [1.0 / self.MAX_SEQUENCE_LENGTH]])

        x1 = self.X1[part_index][sample_idx:sample_idx+1]
        x2 = self.X2[part_index][sample_idx:sample_idx+1]
        x2 += self.MAX_SEQUENCE_LENGTH # make all numbers be positive integers
        x3 = self.X3[part_index][sample_idx:sample_idx+1]
        y1 = self.Y1[part_index][sample_idx:sample_idx+1]
        y2 = self.Y2[part_index][sample_idx:sample_idx+1]
        y3 = self.Y3[part_index][sample_idx:sample_idx+1]
        loss_weights = [[1.0/3.0 for _ in range(3)]] if y1[0][2] == 1 else [[1, 0, 0]]
        loss_weights = np.array(loss_weights)
        return [x1, x2, x3, loss_weights, y1_prev, y2_prev], [y1, y2, y3]


    def train_model(self):
        """
        Train the neural network (ProGlobal) model.
        """
        train_idx = self.loader.part["train"]
        if self.verbose:
            print("self.X1[train_idx].shape", self.X1[train_idx].shape)
            print("self.X2[train_idx].shape", self.X2[train_idx].shape)
            print("self.X3[train_idx].shape", self.X3[train_idx].shape)
            print("self.Y1[train_idx].shape", self.Y1[train_idx].shape)
            print("self.Y2[train_idx].shape", self.Y2[train_idx].shape)
            print("self.Y3[train_idx].shape", self.Y3[train_idx].shape, "\n")

        num_samples = len(self.X1[train_idx])
        cost_history = []
        cost = 0
        print(self.model.metrics_names)
        print("\n### Training starting")
        for epoch in range(self.params.epochs):
            start_time = time.time()
            cost = category_acc = start_acc = end_acc = 0

            y1_prev = y2_prev = None
            print("\nEpoch %d/%d" % (epoch + 1, self.params.epochs))
            sample_idxs = list(range(num_samples))
            random.shuffle(sample_idxs)
            print(sample_idxs[:10])
            for sample_idx in range(num_samples):
                # run training and prediction on sample
                inputs, outputs = self._get_inputs_and_outputs(
                    train_idx, sample_idx, y1_prev, y2_prev)
                metrics = self.model.train_on_batch(x=inputs, y=outputs)
                y1_prev, y2_prev, _ = self.model.predict_on_batch(x=inputs)
                loss = metrics[0]
                cost += loss
                category_acc += metrics[4]
                start_acc += metrics[5]
                end_acc += metrics[6]

                self.write_log(self.callback, self.model.metrics_names, metrics, sample_idx)

                # print current stats
                if sample_idx % 2 == 0 or sample_idx == (num_samples-1):
                    avg_cost = float(cost / (sample_idx + 1))
                    avg_category_acc = float(category_acc / (sample_idx + 1))
                    avg_start_acc = float(start_acc / (sample_idx + 1))
                    avg_end_acc = float(end_acc / (sample_idx + 1))
                    elapsed_time = (time.time() - start_time) / 60.0
                    self.print_progress_bar(sample_idx + 1, num_samples, 
                        "loss: %.3f, avg_cost: %.3f, acc: %.3f, %.3f, %.3f, time (min): %.2f" % (loss, 
                            avg_cost, avg_category_acc, avg_start_acc, avg_end_acc, elapsed_time))

                # save model
                if sample_idx % 100 == 0  or sample_idx == (num_samples-1):
                    self.model.save_weights(self.params.checkpoint_path)
            cost_history.append(cost/num_samples)
        self.model.save(self.params.checkpoint_path)
        print("\n### Training finished")
        print("final cost: ", cost/num_samples)
        print("Model saved to: ", self.params.checkpoint_path)
        return cost_history

    def _get_position(self, y, paragraph_tokens):
        # TODO: get final location
        category = np.argmax(y[0])
        start_pos = np.argmax(y[1])
        end_pos = np.argmax(y[2])
        p_len = len(paragraph_tokens)
        if category == 0:
            return "-"
        if category == 1 or start_pos >= p_len:
            return "?"
        if end_pos < start_pos:
            end_pos = start_pos
        if end_pos > start_pos + 2:
            end_pos = min(start_pos + 2, p_len-1)
        return " ".join(paragraph_tokens[start_pos: end_pos+1])

    def _update_output_grid(self, output_grid, data, sample_idx, y_cur, y_prev):
        """
        Output format follows evaluation from https://arxiv.org/pdf/1808.10012.pdf:
        PID SID PARTICIPANT CHANGE FROM_LOC TO_LOC
        where "CHANGE" can be ("NONE","MOVE","DESTROY","CREATE")
        """
        test_idx = self.loader.part["test"]
        ph_idx, p, s = self.sample_idx_map[test_idx][sample_idx]
        paragraph = data[ph_idx]
        pid = paragraph["PID"]
        participant = paragraph["participants"][p]
        change = "NONE"
        paragraph_tokens = self.ph_idx_tokens_map[test_idx][ph_idx]
        from_loc = self._get_position(y_prev, paragraph_tokens)
        to_loc = self._get_position(y_cur, paragraph_tokens)
        if from_loc != to_loc:
            if to_loc == "-":
                change = "DESTROY"
            elif from_loc == "-":
                change = "CREATE"
            else:
                change = "MOVE"
        output_grid.append([pid, s, participant, change, from_loc, to_loc])

    def output_test_predictions(self):
        """
        Runs trained model on test data
        Generates the output grid and prints it in a file for later evaluation.
        """
        test_idx = self.loader.part["test"]
        num_samples = len(self.X1[test_idx])
        data = self.loader.data[test_idx]
        cost = 0
        start_time = time.time()

        # record previous outputs
        y1_prev = y2_prev = y3_prev = None
        output_grid = []
        print("\n### Test starting")
        for sample_idx in range(num_samples):

            # run prediction on test sample
            inputs, outputs = self._get_inputs_and_outputs(
                test_idx, sample_idx, y1_prev, y2_prev)
            loss = self.model.test_on_batch(x=inputs, y=outputs)
            y_cur = self.model.predict_on_batch(x=inputs)
            self._update_output_grid(output_grid, data, sample_idx, 
                y_cur, y_prev = [y1_prev, y2_prev, y3_prev])
            y1_prev, y2_prev, y3_prev = y_cur

            # print current stats
            cost += loss[0]
            if sample_idx % 10 == 0 or sample_idx == (num_samples-1):
                avg_cost = float(cost / (sample_idx + 1))
                elapsed_time = (time.time() - start_time) / 60.0
                self.print_progress_bar(sample_idx + 1, num_samples, 
                    "cost: %.3f, elapsed time (min): %.2f" % (avg_cost, elapsed_time))
        final_cost = cost/num_samples
        print("\n### Test finished")
        print("final cost: ", final_cost)
        return output_grid, final_cost