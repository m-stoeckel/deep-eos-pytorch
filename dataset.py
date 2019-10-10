import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, Subset
from tqdm import tqdm


class EosDataset(Dataset):
    """
    PyTorch Dataset subclass for deep-eos.
    """

    def __init__(self, train_path, split_dev=True, window_size=5, min_freq=1, save_vocab: Path = None,
                 load_vocab: Path = None, shuffle_input=True, shuffle_dev=True):
        """

        :param train_path:
        :param window_size:
        :param min_freq:c
        :param save_vocab:
        :param load_vocab:
        """
        super(EosDataset, self).__init__()
        with open(train_path, 'r', encoding='utf8') as f:
            training_corpus = f.read()

        if shuffle_input:
            split = training_corpus.split("\n")
            np.random.shuffle(split)
            training_corpus = "\n".join(split)

        data_set_char = self.build_data_set_char(training_corpus, window_size)
        print(data_set_char[:5], data_set_char[-5:])

        if load_vocab is None:
            self.char_2_id_dict = self.build_char_2_id_dict(data_set_char, min_freq)

            if save_vocab is not None:
                self.vocab_size = len(self.char_2_id_dict)
                print(f"Vocabulary size: {self.vocab_size}")
                self.save_vocab(save_vocab)
        else:
            self.load_vocab(load_vocab)

        self.data = self.build_data_set(data_set_char, self.char_2_id_dict, window_size)
        if split_dev:
            subset_len = int(len(self.data) / 10) * 9
            indices = np.arange(len(self.data))
            if shuffle_dev:
                np.random.shuffle(indices)
            self.train = Subset(self, indices[:subset_len])
            self.dev = Subset(self, indices[subset_len:])
        else:
            self.train = self.data
            self.dev = None

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def build_char_2_id_dict(data_set_char, min_freq):
        """
        Builds a char_to_id dictionary

        This methods builds a frequency list of all chars in the data set.

        Then every char gets an own and unique index. Notice: the 0 is reserved
        for unknown chars later, so id labelling starts at 1.

        Author: Stefan Schweter

        :param data_set_char: The input data set (consisting of char sequences)
        :param min_freq: Defines the minimum frequecy a char must appear in data set
        :return: char_2_id dictionary
        """
        char_freq = defaultdict(int)
        char_2_id_table = {}

        for char in [char for label, seq in data_set_char for char in seq]:
            char_freq[char] += 1

        id_counter = 1

        for k, v in [(k, v) for k, v in char_freq.items() if v >= min_freq]:
            char_2_id_table[k] = id_counter
            id_counter += 1

        return char_2_id_table

    @staticmethod
    def build_data_set(data_set_char, char_2_id_dict, window_size):
        """
        Builds a "real" data set with numpy compatible feature vectors

        This method converts the data_set_char to real numpy compatible feature
        vectors. It does also length checks of incoming and outgoing feature
        vectors to make sure that the exact window size is kept

        Author: Stefan Schweter

        :param data_set_char: The input data set (consisting of char sequences)
        :param char_2_id_dict: The char_to_id dictionary
        :param window_size: The window size for the current model
        :return: A data set which contains numpy compatible feature vectors
        """

        data_set = []

        for label, char_sequence in tqdm(data_set_char, desc="Building dataset"):
            ids = []

            if len(char_sequence) == 2 * window_size + 1:
                for char in char_sequence:
                    if char in char_2_id_dict:
                        ids.append(char_2_id_dict[char])
                    else:
                        ids.append(0)

                feature_vector = np.array([float(ids[i])
                                           for i in range(0, len(ids))], dtype=float)

                data_set.append((float(label), feature_vector))

        return data_set

    @staticmethod
    def build_data_set_char(t, window_size):
        """
        Builds data set from corpus

        This method builds a dataset from the training corpus

        Author: Stefan Schweter

        :param t: Input text
        :param window_size: The window size for the current model
        :return: A data set which contains char sequences as feature vectors
        """

        data_set_char_eos = \
            [(1.0, t[m.start() - window_size:m.start()].replace("\n", " ") +
              t[m.start():m.start() + window_size + 1].replace("\n", " "))
             for m in re.finditer('[.:?!;”“"»)][^\n]?[\n]', t)]

        data_set_char_neos = \
            [(0.0, t[m.start() - window_size:m.start()].replace("\n", " ") +
              t[m.start():m.start() + window_size + 1].replace("\n", " "))
             for m in re.finditer('[.:?!;”“"»)][^\s]?[ ]+', t)]

        return data_set_char_eos + data_set_char_neos

    @staticmethod
    def build_potential_eos_list(t, window_size):
        """
        epBuilds a list of potential eos from a given text

        This method builds a list of potential end-of-sentence positions from
        a given text.

        Author: Stefan Schweter

        :param t: Input text
        :param window_size: The window size for the current model
        :return: A list of a pair, like:
            [(1.0, "eht Iv")]
          So the first position in the pair indicates the start position for a
          potential eos. The second position holds the extracted character sequence.
        """

        PUNCT = '[\(\)\u0093\u0094`“”\"›〈⟨〈<‹»«‘’–\'``'']*'
        EOS = r'([.:?!;])'

        eos_positions = [(m.start())
                         for m in re.finditer(r'([.:?!;”“"»)])(\s+' + PUNCT + '|' +
                                              PUNCT + '\s+|[\s\n]+)', t)]

        # Lets extract 2* window_size before and after eos position and remove
        # punctuation

        potential_eos_position = []

        for eos_position in eos_positions:
            left_context = t[eos_position - (2 * window_size):eos_position]
            right_context = t[eos_position:eos_position + (3 * window_size)]

            cleaned_left_context = left_context
            cleaned_right_context = right_context

            # cleaned_left_context = re.sub(PUNCT, '', left_context)
            # cleaned_right_context = re.sub(PUNCT, '', right_context)

            # Also replace multiple whitespaces (use *only* one whitespace)
            cleaned_left_context = re.sub('\s+', ' ', cleaned_left_context)
            cleaned_right_context = re.sub('\s+', ' ', cleaned_right_context)

            potential_eos_position.append((eos_position,
                                           cleaned_left_context[-window_size:] + t[eos_position] +
                                           cleaned_right_context[1:window_size + 1]))

        return potential_eos_position

    def save_vocab(self, vocab_filename) -> None:
        """
        Saves vocabulary to a file

        Author: Stefan Schweter

        :param vocab_filename: The output filename
        :return: None
        """
        with open(vocab_filename, 'wb') as f:
            pickle.dump(self.char_2_id_dict, f, pickle.HIGHEST_PROTOCOL)

    def load_vocab(self, vocab_filename) -> None:
        """
        Loads vocabulary from file

        Author: Stefan Schweter

        :param vocab_filename: The vocabulary filename to be read in
        :return: None
        """
        with open(vocab_filename, 'rb') as f:
            self.char_2_id_dict = pickle.load(f)
