import glob
from pathlib import Path
from typing import Union

import torch
from torch import optim as optim

from dataset import EosDataset
from model import DeepEosModel, train, evaluate


def train_multi():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    window_size = 6
    model_name = 'multi.SETIMES2'

    print("### Loading data ###")
    print("Train data")
    train_data = EosDataset(
        glob.glob('data/SETIMES2.*.train'), shuffle_input=False, split_dev=False,
        save_vocab=Path('multi.SETIMES2/').joinpath(model_name + '.vocab'),
        window_size=window_size, min_freq=10
    )

    print("Dev data")
    dev_data = EosDataset(
        glob.glob('data/SETIMES2.*.dev'), shuffle_input=False, split_dev=False,
        load_vocab=Path('multi.SETIMES2/').joinpath(model_name + '.vocab'),
        window_size=window_size
    )

    model = DeepEosModel(rnn_bidirectional=True, dropout=0.2).to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("### Training ###")
    model = train(model, train_data, dev_data, batch_size=32, epochs=5,
                  optimizer=optimizer, device=device, base_path='multi.SETIMES2/')

    return model


def train_leipzig():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    window_size = 6

    model_name = 'de.leipzig'
    leipzig = EosDataset(
        'data/deu_wikipedia_2016_100K-formatted.txt',
        split_dev=True, save_vocab=Path('leipzig/').joinpath(model_name + '.vocab'),
        window_size=window_size, min_freq=10
    )
    train_data, dev_data = leipzig.train, leipzig.dev

    model = DeepEosModel(rnn_bidirectional=True, dropout=0.2)
    model.to(device)
    print(model)

    # Pre-train on Leipzig corpus
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, train_data, dev_data, batch_size=32, epochs=5,
          optimizer=optimizer, device=device, base_path='leipzig/')

    return model


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
    torch.save(model, Path('biofid_models/').joinpath(model_name + '.h5'))


def temp():
    path = Path('multi.SETIMES2')
    path.mkdir(exist_ok=True)
    model = train_multi()
    # model = str(path.joinpath('best_model.pt'))
    # model = DeepEosModel.load(model).cuda()
    # print(model)
    vocab_path = Path('multi.SETIMES2/').joinpath('multi.SETIMES2.vocab')

    evaluate_all(model, vocab_path)

    # test_data = EosDataset(
    #     glob.glob('data/SETIMES2.*.test'), shuffle_input=False, split_dev=False,
    #     load_vocab=vocab_path,
    #     window_size=6
    # )
    # fine_tune(model, vocab_path, test_data)


def evaluate_all(model, vocab_path):
    print("### Evaluation ###")

    test_data = EosDataset(
        'data/deu_wikipedia_2016_10K-formatted.txt', shuffle_input=True, split_dev=False,
        load_vocab=vocab_path, window_size=6, verbose=False
    )
    print("### Evaluating Leipzig Wiki Corpus ###")
    evaluate(model, test_data, verbose=False)

    test_data = EosDataset(
        'data/deu_mixed-typical_2011_30K-formatted.txt', shuffle_input=True, split_dev=False,
        load_vocab=vocab_path, window_size=6, verbose=False
    )
    print("### Evaluating Leipzig Mixed-Typical Corpus ###")
    evaluate(model, test_data, verbose=False)

    print("### Evaluating SETimes2 Corpus ###")
    for path in glob.glob('data/SETIMES2.*.test'):
        print(path)
        test_data = EosDataset(
            path, shuffle_input=False, split_dev=False,
            load_vocab=vocab_path, window_size=6, verbose=False
        )
        print(path)
        evaluate(model, test_data, verbose=False)

    print("### Evaluating EUROPARL Corpus ###")
    for path in glob.glob('data/europarl-v7.*.test'):
        print(path)
        test_data = EosDataset(
            path, shuffle_input=False, split_dev=False,
            load_vocab=vocab_path, window_size=6, verbose=False
        )
        print(path)
        evaluate(model, test_data, verbose=False)


if __name__ == '__main__':
    temp()
    # model = str(Path('multi.SETIMES2').joinpath('best_model.pt'))
    # model = DeepEosModel.load(model).cuda()
    # print(model)
    #
    # print(tag(model, 'data/plain.txt', vocabulary_path='multi.SETIMES2/multi.SETIMES2.vocab', window_size=6))
