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
    sentences and state changes.
    """

    def __init__(self, data_fname="propara_dataset.tsv", split_fname="propara_split.tsv"):
        self.split_fname = split_fname
        self.data_fname = data_fname 
        self.part = {"train": 0, "dev": 1, "test": 2}
        self.data = [[], [], []]

    def print_stats(self):
        # visualize output label distribution on training data (after under-sampling)

        labels = ('none', 'creation', 'destruction', 'movement')
        y_pos = np.arange(len(labels))
        outputs = np.array([pred["y"] for par in self.data[self.part["train"]] for pred in par["predictions"]])
        unique, counts = np.unique(outputs, return_counts=True)

        plt.bar(y_pos, counts, align='center', alpha=0.5)
        plt.xticks(y_pos, labels)
        plt.ylabel('count')
        plt.title('Output label counts')
         
        plt.show()

    def load_data(self):
        indexes = [set(), set(), set()]
        paragraph = {}

        # Read file containing train, dev, test split of paragraphs.
        with open(self.split_fname, newline='') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                for p, idx in self.part.items():
                    if row[0] == p:
                        indexes[idx].add(row[1])

        # Read paragraph and state change table data. 
        # Compute all the state changes given a sentence and a participant.
        with open(self.data_fname, newline='') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                row = list(map(lambda s: s.lower(), filter(None, row)))
                if len(row) > 0:
                    if row[1] == "sid":
                        paragraph = {"SID": row[0], "participants": row[3:], "states": [], 
                                     "sentences": [], "predictions": []}
                    if row[1].startswith("state"):
                        paragraph["states"].append(row[2:])
                    if row[1].startswith("event"):
                        paragraph["sentences"].append(row[2])
                elif len(paragraph) > 0:
                    for p_name, p_idx in self.part.items():
                        if paragraph["SID"] in indexes[p_idx]:
                            for s in range(len(paragraph["sentences"])):
                                for p in range(len(paragraph["participants"])):    
                                    sentence = paragraph["sentences"][s]
                                    participant = paragraph["participants"][p]
                                    prediction = {"x": {"s": sentence, "p": participant}}
                                    if paragraph["states"][s][p] == "-" and paragraph["states"][s+1][p] != "-":
                                        prediction["y"] = 1 # creation event
                                    elif paragraph["states"][s][p] != "-" and paragraph["states"][s+1][p] == "-":
                                        prediction["y"] = 2 # destruction event
                                    elif paragraph["states"][s][p] != paragraph["states"][s+1][p]:
                                        prediction["y"] = 3 # movement event
                                    else:
                                        prediction["y"] = 0 # nothing happened
                                    # under-sampling "nothing happened" isntances in training
                                    if not (prediction["y"] == 0 and p_name == "train" and random.randint(0,2) != 0):
                                        paragraph["predictions"].append(prediction)
                            self.data[p_idx].append(paragraph)
                            break


def main():
    # Debug only.
    loader = Loader()
    loader.load_data()
    loader.print_stats()

if __name__ == "__main__":
    main()