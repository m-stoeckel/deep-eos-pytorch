"""
Dataset class and functions for the PyTorch implementation of DeepEOS.
"""
__author__ = 'Manuel Stoeckel'

import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Union, Iterable, List

import numpy as np
from torch.utils.data import Dataset, Subset
from tqdm import tqdm


class EosDataset(Dataset):
    """
    PyTorch Dataset subclass for deep-eos.
    """

    def __init__(self, train_path: Union[str, Path], split_dev=True, window_size=5, min_freq=1,
                 save_vocab: Path = None, load_vocab: Path = None, shuffle_input=True, shuffle_dev=True,
                 use_default_markers=True, remove_duplicates=True, verbose=True):
        """
        PyTorch Dataset subclass for deep-eos.

        :param train_path: A single or a list of file paths to load as training data.
        :param split_dev: If True, split 10% of the corpus from the training data as development data.
        :param window_size: The window size around EOS characters.
        :param min_freq: The minimum frequency of characters to be considered in the given window.
        :param save_vocab: If a Path is given, the vocabulary will be saved here.
        :param load_vocab: If a Path is given, the vocabulary will be loaded from here.
        :param shuffle_input: If True, shuffle the input corpus lines prior to EOS extraction.
        :param shuffle_dev: If True, return a random subsample instead of the last 10%.
        :param use_default_markers: If False, use extended EOS markers including more characters.
        :param remove_duplicates: If True, remove all duplicates samples.
        :param verbose: If False, disable tqdm progress bars.
        """
        super(EosDataset, self).__init__()

        data_set_char = self.get_char_data(train_path, shuffle_input, window_size, use_default_markers, verbose)

        if remove_duplicates:
            data_set_char = list(dict.fromkeys(data_set_char))

        if load_vocab is None:
            self.char_2_id_dict = self.build_char_2_id_dict(data_set_char, min_freq, verbose)

            if save_vocab is not None:
                self.vocab_size = len(self.char_2_id_dict)
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

    def get_char_data(self, train_path, shuffle_input, window_size, use_default_markers, verbose):
        with tqdm(desc="Loading corpus", total=1, ascii=True, disable=not verbose) as tq:
            with open(train_path, 'r', encoding='utf8') as f:
                training_corpus = f.read()

            if shuffle_input:
                training_corpus = self.shuffle(training_corpus)

            data_set_char = self.build_data_set_char(training_corpus, window_size, use_default_markers)
            tq.update()
        return data_set_char

    @staticmethod
    def shuffle(training_corpus):
        split = training_corpus.split("\n")
        np.random.shuffle(split)
        training_corpus = "\n".join(split)
        return training_corpus

    @staticmethod
    def build_char_2_id_dict(data_set_char, min_freq, verbose=True):
        """
        Builds a char_to_id dictionary

        This methods builds a frequency list of all chars in the data set.

        Then every char gets an own and unique index. Notice: the 0 is reserved
        for unknown chars later, so id labelling starts at 1.

        Author: Stefan Schweter

        :param data_set_char: The input data set (consisting of char sequences)
        :param min_freq: Defines the minimum frequecy a char must appear in data set
        :param verbose: If False, disable tqdm progress bars.
        :return: char_2_id dictionary
        """
        char_freq = defaultdict(int)
        char_2_id_table = {}

        chars = [char for label, seq in data_set_char for char in seq]
        for char in tqdm(chars, desc="Building vocabulary", ascii=True, disable=not verbose):
            char_freq[char] += 1

        id_counter = 1

        for k, v in [(k, v) for k, v in char_freq.items() if v >= min_freq]:
            char_2_id_table[k] = id_counter
            id_counter += 1

        print(f"Vocabulary size: {len(char_2_id_table)}")

        return char_2_id_table

    @staticmethod
    def build_data_set(data_set_char, char_2_id_dict, window_size, verbose=True):
        """
        Builds a "real" data set with numpy compatible feature vectors

        This method converts the data_set_char to real numpy compatible feature
        vectors. It does also length checks of incoming and outgoing feature
        vectors to make sure that the exact window size is kept

        Author: Stefan Schweter

        :param data_set_char: The input data set (consisting of char sequences)
        :param char_2_id_dict: The char_to_id dictionary
        :param window_size: The window size for the current model
        :param verbose: If False, disable tqdm progress bars.
        :return: A data set which contains numpy compatible feature vectors
        """

        data_set = []

        for label, char_sequence in tqdm(data_set_char, desc="Building dataset", ascii=True, disable=not verbose):
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
    def build_data_set_char(t, window_size, use_default_marker=True):
        """
        Builds data set from corpus

        This method builds a dataset from the training corpus

        Author: Stefan Schweter

        :param t: Input text
        :param window_size: The window size for the current model
        :param use_default_marker: If false, use expanded EOS marker definition
        :return: A data set which contains char sequences as feature vectors
        """
        eos = r'.:?!;' if use_default_marker else r'.:?!;”“"»'

        data_set_char_eos = \
            [(1.0, t[m.start() - window_size:m.start()].replace("\n", " ") +
              t[m.start():m.start() + window_size + 1].replace("\n", " "))
             for m in re.finditer(f'[{eos}][^\n]?[\n]', t)]

        data_set_char_neos = \
            [(0.0, t[m.start() - window_size:m.start()].replace("\n", " ") +
              t[m.start():m.start() + window_size + 1].replace("\n", " "))
             for m in re.finditer(f'[{eos}][^\\s]?[ ]+', t)]

        return data_set_char_eos + data_set_char_neos

    @staticmethod
    def build_potential_eos_list(t, window_size, use_default_markers=True):
        """
        epBuilds a list of potential eos from a given text

        This method builds a list of potential end-of-sentence positions from
        a given text.

        Author: Stefan Schweter

        :param t: Input text
        :param window_size: The window size for the current model
        :param use_default_markers: If false, use expanded EOS marker definition
        :return: A list of a pair, like:
            [(1.0, "eht Iv")]
          So the first position in the pair indicates the start position for a
          potential eos. The second position holds the extracted character sequence.
        """

        punct = '[()\u0093\u0094`“”\"›〈⟨〈<‹»«‘’–\'``'']*'
        eos = r'.:?!;' if use_default_markers else r'.:?!;”“"»'

        eos_positions = [(m.start())
                         for m in re.finditer(f'([{eos}])(\\s+' + punct + '|' + punct + '\\s+|[\\s\n]+)', t)]

        # Lets extract 2* window_size before and after eos position and remove
        # punctuation

        potential_eos_position = []

        for eos_position in eos_positions:
            left_context = t[eos_position - (2 * window_size):eos_position]
            right_context = t[eos_position:eos_position + (3 * window_size)]

            cleaned_left_context = left_context
            cleaned_right_context = right_context

            # cleaned_left_context = re.sub(punct, '', left_context)
            # cleaned_right_context = re.sub(punct, '', right_context)

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


class EosMultiDataset(EosDataset):
    def __init__(self, train_path: Iterable[Union[str, Path]], *args, **kwargs):
        super(EosMultiDataset, self).__init__(train_path, *args, **kwargs)

    def get_char_data(self, train_path: Iterable[Union[str, Path]], shuffle_input, window_size, use_default_markers,
                      verbose):
        data_set_char = []
        for path in tqdm(train_path, desc="Loading corpora", ascii=True, disable=not verbose):
            with open(path, 'r', encoding='utf8') as f:
                training_corpus = f.read()

            if shuffle_input:
                training_corpus = self.shuffle(training_corpus)

            data_set_char.extend(self.build_data_set_char(training_corpus, window_size, use_default_markers))

        return data_set_char


class ListDataset(Dataset):
    def __init__(self, input: List[str]):
        super(ListDataset, self).__init__()
        self.data = input

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)
