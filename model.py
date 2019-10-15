"""
Model class and functions for the PyTorch implementation of DeepEOS.
"""
import pickle

import numpy as np

__author__ = 'Manuel Stoeckel'

from pathlib import Path
from typing import Union, List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EosDataset, ListDataset
from util import AverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DeepEosModel(nn.Module):
    dense: nn.Linear
    _dropout: nn.Dropout
    rnn: Union[nn.GRU, nn.LSTM]
    emb: nn.Embedding

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

        # Build model from hyper-parameters
        self.build()

    def build(self):
        self.emb = nn.Embedding(num_embeddings=self.max_features,
                                embedding_dim=self.embdding_size)
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.embdding_size,
                               hidden_size=self.rnn_size,
                               num_layers=self.rnn_layers,
                               dropout=self.dropout if self.rnn_layers > 1 else 0.0,
                               bidirectional=self.rnn_bidirectional,
                               batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.embdding_size,
                              hidden_size=self.rnn_size,
                              num_layers=self.rnn_layers,
                              dropout=self.dropout if self.rnn_layers > 1 else 0.0,
                              bidirectional=self.rnn_bidirectional,
                              batch_first=True)
        else:
            raise NotImplementedError(f"'{self.rnn_type}' is not a valid RNN type choice. Please use: [LSTM, GRU]")
        self._dropout = nn.Dropout(p=self.dropout)
        self.dense = nn.Linear(in_features=self.rnn_size * (2 if self.rnn_bidirectional else 1),
                               out_features=1)

    def forward(self, input: torch.Tensor):
        output = self.emb(input)
        self.rnn.flatten_parameters()
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

    def load(self, model_path: Union[Path, str]):
        if type(model_path) is str:
            model_path = Path(model_path)
        model_dict = torch.load(model_path)
        for attr, value in model_dict['hyper_params'].items():
            self.__setattr__(attr, value)
        self.build()
        self.load_state_dict(model_dict['state_dict'])
        return self

    @staticmethod
    def from_file(model_path: Union[Path, str]) -> nn.Module:
        """
        Create a new DeepEosModel from a model saved using the checkpoint() method.

        :param model_path: The file path of the saved model.
        :return: A new DeepEosModel.
        """
        if type(model_path) is str:
            model_path = Path(model_path)

        model_dict = torch.load(model_path)
        model = DeepEosModel(**model_dict['hyper_params'])
        model.load_state_dict(model_dict['state_dict'])
        return model


class DeepEosDataParallel(nn.DataParallel):
    """
    DeepEOS wrapper class for nn.DataParallel
    """

    def __init__(self, module: DeepEosModel):
        super(DeepEosDataParallel, self).__init__(module)

    def __getattr__(self, name):
        try:
            return super(DeepEosDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

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
            'state_dict': self.module.state_dict()
        }
        torch.save(model_dict, str(model_path), pickle_protocol=4)

    def load(self, model_path: Union[Path, str]):
        self.module.load(model_path)

    @staticmethod
    def from_file(model_path: Union[Path, str]) -> nn.Module:
        """
        Create a new DeepEosModel from a model saved using the checkpoint() method.

        :param model_path: The file path of the saved model.
        :return: A new DeepEosModel.
        """
        if type(model_path) is str:
            model_path = Path(model_path)

        model_dict = torch.load(model_path)
        model = DeepEosModel(**model_dict['hyper_params'])
        model.load_state_dict(model_dict['state_dict'])
        return DeepEosDataParallel(model)


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
        loss_meter = AverageMeter()
        score_meter = AverageMeter()
        with tqdm(train_loader, total=total_batches, desc=f"Epoch {epoch + 1}", ascii=True, miniters=10) as tq:
            for batch_no, (y_train, x_train) in enumerate(tq):
                y_train = y_train.float().to(device)
                x_train = x_train.long().to(device)

                prediction = model(x_train).squeeze()

                optimizer.zero_grad()
                loss = criterion(prediction, y_train)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    y_eval = y_train.bool().cpu().detach().tolist()
                    pred_eval = (prediction >= 0.5).cpu().detach().tolist()
                    if eval_metric is 'precision':
                        score = precision_score(y_eval, pred_eval, pos_label=True)
                    elif eval_metric is 'recall':
                        score = recall_score(y_eval, pred_eval, pos_label=True)
                    else:
                        score = f1_score(y_eval, pred_eval, pos_label=True)

                loss_meter.update(loss.item())
                score_meter.update(score)
                tq.set_postfix_str(f"loss: {loss_meter.avg:0.4f} ({loss.item():0.4f}), "
                                   f"{eval_metric}: {score_meter.avg:0.4f} ({score:0.4f})", True)

        if evaluate_after_epoch and dev_dataset is not None:
            print("Development dataset - ", end="")
            score = get_score(model, dev_dataset, batch_size=eval_batch_size, metric=eval_metric, device=device,
                              verbose=False)

            if save_checkpoints and base_path is not None and score > best_score:
                best_score = score
                model_file = base_path / "best_model.pt"
                DeepEosModel.checkpoint(model, model_file)

        if save_checkpoints and base_path is not None:
            model_file = base_path / "last_model.pt"
            DeepEosModel.checkpoint(model, model_file)

    if epochs > 1 and evaluate_after_epoch and dev_dataset is not None and save_checkpoints and base_path is not None:
        print(
            "Loading best scoring model\n"
            f"{eval_metric.title()}: {best_score:0.4f}"
        )
        return DeepEosModel.from_file(str(base_path / "best_model.pt")).to(device)
    return model


def evaluate(model: DeepEosModel, dataset: Union[EosDataset, list], batch_size=32, verbose=True,
             device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> tuple:
    """
    Evaluate the given model.

    :param model: The model to evaluate.
    :param dataset: The evaluation dataset.
    :param batch_size: The evaluation batch size.
    :param metric: The evaluation metric to return.
    :param device: See torch.device.
    :param verbose: If False, disable tqdm progress bars.
    :return:
    """
    model.eval()

    true_samples = []
    pred_samples = []
    dev_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with tqdm(dev_loader, total=len(dev_loader), desc="Evaluating", ascii=True, disable=not verbose) as tq:
        for batch_no, (y_eval, x_eval) in enumerate(tq):
            true_samples.extend(y_eval.squeeze().bool().cpu().tolist())
            x_eval = x_eval.long().to(device)

            prediction = model(x_eval) >= 0.5
            pred_samples.extend(prediction.squeeze().bool().cpu().tolist())

    precision = precision_score(true_samples, pred_samples, pos_label=True)
    recall = recall_score(true_samples, pred_samples, pos_label=True)
    f1 = f1_score(true_samples, pred_samples, pos_label=True)
    print(f"Precision: {precision:0.4f}, "
          f"Recall: {recall:0.4f}, "
          f"F1: {f1:0.4f}", flush=True)
    return (precision, recall, f1)


def get_score(model: DeepEosModel, dataset: Union[EosDataset, list], metric: Union[str, List[str]] = 'precision',
              *args, **kwargs) -> float:
    precision, recall, f1 = evaluate(model, dataset, *args, **kwargs)
    if metric is 'recall':
        return recall
    elif metric is 'f1':
        return f1
    else:
        return precision


def tag(model: DeepEosModel, text_file: Union[str, Path], vocabulary_path: Union[str, Path], batch_size=32,
        window_size=5, return_indices=False, use_default_markers=True,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    with open(vocabulary_path, 'rb') as f:
        vocabulary = pickle.load(f)

    peos_list = []
    with open(text_file, 'r', encoding='utf8') as f:
        text = f.read()
        peos_list.extend(EosDataset.build_potential_eos_list(text, window_size, use_default_markers))
        text = np.asarray(list(text), dtype=str)

    dataset = ListDataset(EosDataset.build_data_set(peos_list, vocabulary, window_size))

    ret_idx = []
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    with tqdm(dataloader, total=len(dataloader), desc="Tagging", ascii=True) as tq:
        for batch_no, (indices, x_tag) in enumerate(tq):
            x_tag = x_tag.long().to(device)
            prediction = model(x_tag) >= 0.5

            zipped = zip(indices.squeeze().int().cpu().tolist(), prediction.squeeze().cpu().tolist())
            true_indices = [idx + 1 for idx, _ in filter(lambda p: p[1], zipped)]
            if return_indices:
                ret_idx.extend(true_indices)
            else:
                text[np.array(true_indices)] = '\n'

    if return_indices:
        return ret_idx
    else:
        return "".join(text)
