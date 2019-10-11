"""
Model class and functions for the PyTorch implementation of DeepEOS.
"""
__author__ = 'Manuel Stoeckel'

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


class DeepEosModel(nn.Module):
    def __init__(self, max_features=1000, embedding_size=128, rnn_type: str = 'LSTM', rnn_size=256,
                 dropout=0.2, rnn_layers=1, rnn_bidirectional=False):
        super(DeepEosModel, self).__init__()

        # Hyperparameters
        self.max_features = max_features
        self.embdding_size = embedding_size
        self.rnn_type = rnn_type.lower()
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.rnn_layers = rnn_layers
        self.rnn_bidirectional = rnn_bidirectional

        # Layers
        self.emb = nn.Embedding(num_embeddings=self.max_features,
                                embedding_dim=self.embdding_size)

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.embdding_size,
                               hidden_size=self.rnn_size,
                               num_layers=self.rnn_layers,
                               dropout=self.dropout if self.rnn_layers > 1 else 0.0,
                               bidirectional=self.rnn_bidirectional,
                               batch_first=True)
        elif str(rnn_type).lower() == 'gru':
            self.rnn = nn.GRU(input_size=self.embdding_size,
                              hidden_size=self.rnn_size,
                              num_layers=self.rnn_layers,
                              dropout=self.dropout if self.rnn_layers > 1 else 0.0,
                              bidirectional=self.rnn_bidirectional,
                              batch_first=True)
        else:
            raise NotImplementedError(f"'{rnn_type}' is not a valid RNN type choice. Please use: [LSTM, GRU]\n")

        self._dropout = nn.Dropout(p=self.dropout)
        self.dense = nn.Linear(in_features=self.rnn_size * (2 if self.rnn_bidirectional else 1),
                               out_features=1)

    def forward(self, input: torch.Tensor):
        output = self.emb(input)
        output, _ = self.rnn(output)
        output = self._dropout(output[:, -1, :].squeeze())
        output = self.dense(output)
        return torch.sigmoid(output)

    def checkpoint(self, model_path: Union[str, Path]):
        """
        Create a checkpoint file a the given path.

        :param model_path: The file path for the new checkpoint.
        :return: None
        """
        model_dict = {
            'hyper_params': {
                'max_features': self.max_features,
                'embedding_size': self.embdding_size,
                'rnn_size': self.rnn_size,
                'rnn_layers': self.rnn_layers,
                'rnn_type': self.rnn_type,
                'dropout': self.dropout,
                'rnn_bidirectional': self.rnn_bidirectional
            },
            'state_dict': self.state_dict()
        }
        torch.save(model_dict, str(model_path), pickle_protocol=4)

    @staticmethod
    def load(model_path: Union[Path, str]):
        if type(model_path) is str:
            model_path = Path(model_path)

        model_dict = torch.load(model_path)
        model = DeepEosModel(**model_dict['hyper_params'])
        model.load_state_dict(model_dict['state_dict'])
        return model


def train(model: DeepEosModel, train_dataset, dev_dataset=None, optimizer=None, epochs=5, batch_size=32,
          evaluate_after_epoch=True, eval_batch_size=32, base_path: Union[str, Path] = None, save_checkpoints=True,
          eval_metric='precision',
          device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> DeepEosModel:
    """
    Train the given DeepEosModel with the given datasets using Binary-Cross-Entropy loss.

    :param model: The model to train.
    :param train_dataset: The train dataset.
    :param dev_dataset: The dev dataset.
    :param optimizer: The optimizer, defaults to Adam with lr=0.001.
    :param epochs: The number of training epochs.
    :param batch_size: The batch size.
    :param evaluate_after_epoch: If true, evaluate after each batch using the dev dataset.
    :param eval_batch_size: The batch size for the evaluation.
    :param base_path: The base path for model checkpoints.
    :param save_checkpoints: If True, save checkpoints after each epoch in the given base path.
     Will save the last and the best model in separate files.
    :param eval_metric: The evaluation metric to choose the best model.
    :param device: See torch.device.
    :return: The best model if checkpointing and dev evaluation were enabled. The last model otherwise.
    """
    criterion = nn.BCELoss()
    criterion.to(device)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    if type(base_path) is str:
        base_path = Path(base_path)
    best_score = -1.0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_batches = len(train_loader)
        average = AverageMeter()
        with tqdm(train_loader, total=total_batches, desc=f"Epoch {epoch + 1}") as tq:
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

        if evaluate_after_epoch and dev_dataset is not None:
            score = evaluate(model, dev_dataset, eval_batch_size, metric=eval_metric, device=device)

            if save_checkpoints and base_path is not None and score > best_score:
                best_score = score
                model_file = base_path / "best_model.pt"
                model.checkpoint(model_file)

        if save_checkpoints and base_path is not None:
            model_file = base_path / "last_model.pt"
            model.checkpoint(model_file)

    if epochs > 1 and evaluate_after_epoch and dev_dataset is not None and save_checkpoints and base_path is not None:
        print(
            "Loading best scoring model\n"
            f"{eval_metric.title()}: {best_score * 100:0.2f}%"
        )
        return DeepEosModel.load(str(base_path / "best_model.pt")).to(device)
    return model


def evaluate(model: DeepEosModel, datset: Union[EosDataset, list], batch_size=32, metric='precision',
             device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> float:
    """
    Evaluate the given model.

    :param model: The model to evaluate.
    :param datset: The evaluation dataset.
    :param batch_size: The evaluation batch size.
    :param metric: The evaluation metric to return.
    :param device: See torch.device.
    :return:
    """
    model.eval()

    true_samples = []
    pred_samples = []
    dev_loader = DataLoader(datset, batch_size=batch_size, shuffle=True)
    with tqdm(dev_loader, total=len(dev_loader), desc="Evaluating", position=0) as tq:
        for batch_no, (y_eval, x_eval) in enumerate(tq):
            true_samples.extend(y_eval.squeeze().bool().cpu().tolist())
            x_eval = x_eval.long().to(device)

            prediction = model(x_eval) >= 0.5
            pred_samples.extend(prediction.squeeze().bool().cpu().tolist())

    precision = precision_score(true_samples, pred_samples, pos_label=True)
    recall = recall_score(true_samples, pred_samples, pos_label=True)
    f1 = f1_score(true_samples, pred_samples, pos_label=True)
    print(f"Precision: {precision * 100:0.2f}%, "
          f"Recall: {recall * 100:0.2f}%, "
          f"F1: {f1 * 100:0.2f}%", flush=True)
    if metric is 'recall':
        return recall
    elif metric is 'f1':
        return f1
    else:
        return precision
