# imports and dependencies
import os
import sys
import csv
import spacy
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

#################################################################################
### Input (read and parse corpora)
##################################################################################

class Loader:
    """
    Class for loading and storing dataset information such as paragraphs, 
    sentences and state changes. Also loads glove embeddings.
    """

    def __init__(self, data_fname="propara_dataset.tsv", split_fname="propara_split.tsv", 
                 glove_folder="glove.6B"):
        self.split_fname = split_fname
        self.data_fname = data_fname 
        self.glove_dir = os.path.join(os.getcwd(), glove_folder)
        self.part = {"train": 0, "dev": 1, "test": 2}
        self.data = [[], [], []]
        self.embeddings_index = {}

    def print_stats(self):
        # visualize output label distribution on training data (after under-sampling)

        labels = ('does not exist', 'location unknown', 'location known')
        y_pos = np.arange(len(labels))
        outputs = np.array([pred for par in self.data[self.part["train"]] 
            for pred in par["predictions"].values()])
        unique, counts = np.unique(outputs, return_counts=True)

        plt.bar(y_pos, counts, align='center', alpha=0.5)
        plt.xticks(y_pos, labels)
        plt.ylabel('count')
        plt.title('Output label counts')
        plt.show()

        print('Number of word vectors = ', len(self.embeddings_index))

    def load_data(self):
        self.load_propara_data()
        self.load_embeddings()

    def load_propara_data(self):
        self.indexes = [set(), set(), set()]
        paragraph = {}

        # Read file containing train, dev, test split of paragraphs.
        with open(self.split_fname, newline='') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                for p, idx in self.part.items():
                    if row[0] == p:
                        self.indexes[idx].add(row[1])

        # Read paragraph and state change table data.
        # Compute all the state changes given a sentence and a participant.
        with open(self.data_fname, newline='') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                row = list(map(lambda s: s.lower(), filter(None, row)))
                if len(row) > 0:
                    if row[1] == "sid":
                        paragraph = {"PID": row[0], "participants": row[3:], "states": [], 
                                     "sentences": [], "predictions": []}
                    if row[1].startswith("state"):
                        paragraph["states"].append(row[2:])
                    if row[1].startswith("event"):
                        sentence = row[2].replace(",", " ,").replace(".", " .").replace(";", " ;")
                        paragraph["sentences"].append(sentence)
                elif len(paragraph) > 0:
                    for p_name, p_idx in self.part.items():
                        if paragraph["PID"] in self.indexes[p_idx]:
                            predictions = {}
                            for s in range(len(paragraph["sentences"])):
                                for p in range(len(paragraph["participants"])):
                                    key = (s,p)
                                    if paragraph["states"][s+1][p] == "-":
                                        predictions[key] = 1 # Does not exist.
                                    elif paragraph["states"][s+1][p] != "?":
                                        predictions[key] = 2 # Location Unknown.
                                    else:
                                        predictions[key] = 0 # Location is Known.
                            paragraph["predictions"] = predictions
                            self.data[p_idx].append(paragraph)
                            paragraph = {}
                            break

    def load_embeddings(self):
        with open(os.path.join(self.glove_dir, 'glove.6B.50d.txt'),  encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs

def main():
    # Debug only.
    print("loading data...")
    loader = Loader()
    loader.load_data()
    loader.print_stats()

if __name__ == "__main__":
    main()