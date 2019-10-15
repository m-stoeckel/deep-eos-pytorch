import glob
import os
from pathlib import Path
from typing import Union

import torch
from torch import optim as optim

from dataset import EosDataset, EosMultiDataset
from model import DeepEosModel, train, evaluate, DeepEosDataParallel


def fine_tune(model: Union[DeepEosModel, str], vocab_path: Union[str, Path], cross_validation_set=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    window_size = 6

    if type(model) is str:
        model = torch.load(str(model), map_location=device)

    # BIOfid fine-tuning
    biofid_train = EosDataset(
        'data/bioFID_train_cleaned.txt', shuffle_input=False,
        split_dev=False, load_vocab=vocab_path, window_size=window_size
    )
    biofid_test = EosDataset(
        'data/bioFID_test.txt', shuffle_input=False,
        split_dev=False, load_vocab=vocab_path, window_size=window_size
    )

    print("\nBIOfid test scores prior to fine-tuning")
    evaluate(model, biofid_test, device=device)
    print(flush=True)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(model, biofid_train, biofid_test, base_path=Path('biofid_models/'),
          optimizer=optimizer, epochs=10, device=device)

    print("\nBIOfid test")
    evaluate(model, biofid_test, device=device)

    if cross_validation_set is not None:
        print("\nCross validation")
        evaluate(model, cross_validation_set, device=device)

    model_name = 'de.biofid'
    torch.save(model, Path('biofid_models/').joinpath(model_name + '.pt'))


def train_all():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_path = Path("models")
    base_path.mkdir(exist_ok=True)
    files = list(glob.glob('data/europarl-v7.*.train')) + glob.glob('data/SETIMES2.*.train')

    # LSTM
    for train_file in files:
        run_training(train_file, base_path, device, False)
    # BiLSTM
    for train_file in files:
        run_training(train_file, base_path, device, True)


def run_training(train_file, base_path, device, bidirectional=False):
    corpus_name = os.path.split(train_file)[1].replace('.train', '')
    print(f"Training with {corpus_name}")
    model_name = corpus_name + ("_LSTM" if not bidirectional else "_BiLSTM")
    model_path = base_path.joinpath(model_name)
    model_path.mkdir(exist_ok=True)
    dev_file = train_file.replace(".train", ".dev")
    test_file = train_file.replace(".train", ".test")
    print(f"Loading {train_file}")
    train_data = EosDataset(train_file, split_dev=False, min_freq=10000, remove_duplicates=False,
                            save_vocab=model_path.joinpath("vocab.pickle"))
    print(f"Loading {dev_file}")
    dev_data = EosDataset(dev_file, split_dev=False, min_freq=10000, remove_duplicates=False,
                          load_vocab=model_path.joinpath("vocab.pickle"))
    print(f"Loading {test_file}")
    test_data = EosDataset(test_file, split_dev=False, min_freq=10000, remove_duplicates=False,
                           load_vocab=model_path.joinpath("vocab.pickle"))
    model = DeepEosModel(max_features=20000, rnn_bidirectional=bidirectional)
    model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Starting Traning")
    train(model, train_data, dev_data, base_path=model_path, optimizer=optimizer, epochs=5, device=device)
    print("Evaluating")
    with open(model_path.joinpath("evaluation.txt"), 'w', encoding='UTF-8') as f:
        precision, recall, f1, accuracy = evaluate(model, test_data, device=device)
        f.write(f"Precision: {precision}\nRecall: {recall}\nF1: {f1}\nAccuracy: {accuracy}\n")


if __name__ == '__main__':
    train_all()
