from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EosDataset
from util import AverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DeepEos(nn.Module):
    def __init__(self, max_features=1000, embedding_size=128, rnn_type: str = 'LSTM', rnn_size=256,
                 dropout=.25, rnn_layers=1, rnn_bidirectional=False):
        super(DeepEos, self).__init__()

        self.emb = nn.Embedding(num_embeddings=max_features,
                                embedding_dim=embedding_size)
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_size,
                               hidden_size=self.rnn_size,
                               num_layers=self.rnn_layers,
                               dropout=dropout if self.rnn_layers > 1 else 0.0,
                               bidirectional=rnn_bidirectional,
                               batch_first=True)
        elif str(rnn_type).lower() == 'gru':
            self.rnn = nn.GRU(input_size=embedding_size,
                              hidden_size=self.rnn_size,
                              num_layers=self.rnn_layers,
                              dropout=dropout if self.rnn_layers > 1 else 0.0,
                              bidirectional=rnn_bidirectional,
                              batch_first=True)
        else:
            raise NotImplementedError(f"'{rnn_type}' is not a valid RNN type choice. Please use: [LSTM, GRU]\n")

        self.dropout = nn.Dropout(p=dropout)
        self.dense = nn.Linear(in_features=self.rnn_size * (2 if rnn_bidirectional else 1),
                               out_features=1)

    def forward(self, input: torch.Tensor):
        output = self.emb(input)
        output, _ = self.rnn.forward_impl(output, None, None, input.size(0), None)
        output = self.dropout(output[:, -1, :].squeeze())
        output = self.dense(output)
        return torch.sigmoid(output)


def train(model: DeepEos, train_dataset, dev_datset=None, optimizer=None, epochs=5, batch_size=32,
          evaluate_after_epoch=True, eval_batch_size=32,
          device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.BCELoss()
    criterion.to(device)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        model.train()
        total_batches = len(train_loader)
        average = AverageMeter()
        with tqdm(train_loader, total=total_batches, desc="Training") as tq:
            for batch_no, (y_train, x_train) in enumerate(tq):
                x_train = x_train.long().to(device)

                model.rnn.flatten_parameters()
                prediction = model(x_train)

                optimizer.zero_grad()
                loss = criterion(prediction.squeeze(), y_train.float().to(device))
                loss.backward()
                optimizer.step()

                average.update(loss.item())
                tq.set_postfix_str(f"loss: {average.avg:0.5f} ({loss.item():0.5f})", True)

        # Evaluate
        if evaluate_after_epoch and dev_datset is not None:
            evaluate(model, dev_datset, eval_batch_size, device)


def evaluate(model: DeepEos, dev_datset: Union[EosDataset, list], batch_size=32,
             device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    model.eval()

    true_samples = []
    pred_samples = []
    dev_loader = DataLoader(dev_datset, batch_size=batch_size, shuffle=True)
    with tqdm(dev_loader, total=len(dev_loader), desc="Evaluating", position=0) as tq:
        for batch_no, (y_eval, x_eval) in enumerate(tq):
            true_samples.extend(y_eval.squeeze().bool().cpu().tolist())
            x_eval = x_eval.long().to(device)

            prediction = model(x_eval) >= 0.5
            pred_samples.extend(prediction.squeeze().bool().cpu().tolist())

    print(f"Precision: {precision_score(true_samples, pred_samples, pos_label=True) * 100:0.2f}%, "
          f"Recall: {recall_score(true_samples, pred_samples, pos_label=True) * 100:0.2f}%, "
          f"F1: {f1_score(true_samples, pred_samples, pos_label=True) * 100:0.2f}%", flush=True)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    window_size = 6

    model_name = 'de.biofid'
    leipzig = EosDataset(
        Path('data/deu_wikipedia_2016_100K-formatted.txt'),
        split_dev=True, save_vocab=Path('models/').joinpath(model_name + '.vocab'),
        window_size=window_size, min_freq=10
    )
    train_data, dev_data = leipzig.train, leipzig.dev
    # model_name = 'multi.SETIMES2'
    # train_data = EosDataset(
    #     Path('data/SETIMES2.all.train'), shuffle_input=False,
    #     split_dev=False, save_vocab=Path('models/').joinpath(model_name + '.vocab'),
    #     window_size=window_size, min_freq=10
    # )
    # dev_data = EosDataset(
    #     Path('data/SETIMES2.all.dev'), shuffle_input=False,
    #     split_dev=False, load_vocab=Path('models/').joinpath(model_name + '.vocab'),
    #     window_size=window_size, min_freq=10
    # )
    biofid_train = EosDataset(
        Path('data/bioFID_train_cleaned.txt'), shuffle_input=False,
        split_dev=False, load_vocab=Path('models/').joinpath(model_name + '.vocab'),
        window_size=window_size
    )
    biofid_test = EosDataset(
        Path('data/bioFID_test.txt'), shuffle_input=False,
        split_dev=False, load_vocab=Path('models/').joinpath(model_name + '.vocab'),
        window_size=window_size
    )

    model = DeepEos(rnn_bidirectional=True)  # TODO
    # pre-train
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(model, train_data, dev_data, batch_size=32,
          optimizer=optimizer, epochs=1, device=device)
    print("\nBIOfid dev")
    evaluate(model, biofid_test.train, device=device)
    print(flush=True)

    # BIOfid train
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(model, biofid_train.train, biofid_test.train,
          optimizer=optimizer, epochs=1, device=device)

    torch.save(model.state_dict(), Path('models/').joinpath('multi.SETIMES2.h5'))


if __name__ == '__main__':
    main()
