import json
import os
from functools import partial
from collections import defaultdict
from typing import Callable, Dict, Optional

import numpy as np
from tqdm import tqdm
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from sklearn.metrics import f1_score, accuracy_score

from src.utils.metrics import evaluate_tokenwise_metric, evaluate_examplewise_accuracy


def train(model: Module,
          loader: DataLoader,
          optimizer: Optimizer,
          device: object,
          clip: float = 3.,
          last_n_losses: int = 500,
          verbose: bool = True):
    losses = []

    progress_bar = tqdm(total=len(loader), disable=not verbose, desc='Train')

    model.train()

    for encoder_seq, target_seq, true_lens in loader:
        encoder_seq = encoder_seq.to(device)
        target_seq = target_seq.to(device)

        loss = model(encoder_seq, target_seq)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        losses.append(loss.item())

        progress_bar.set_postfix(loss=np.mean(losses[-last_n_losses:]))

        progress_bar.update()

    progress_bar.close()

    return losses


def evaluate(model: Module,
             loader: DataLoader,
             device: object,
             metrics: Optional[Dict[str, Callable]] = None,
             last_n_losses: int = 500,
             verbose: bool = True):
    losses = []
    epoch_scores = dict()
    batch_scores = defaultdict(lambda: [])

    progress_bar = tqdm(total=len(loader), disable=not verbose, desc='Evaluate')

    model.eval()

    for encoder_seq, target_seq, true_lens in loader:
        encoder_seq = encoder_seq.to(device)
        target_seq = target_seq.to(device)

        with torch.no_grad():
            loss = model(encoder_seq, target_seq)
            preds = model.predict(encoder_seq)

        losses.append(loss.item())

        progress_bar.set_postfix(loss=np.mean(losses[-last_n_losses:]))

        progress_bar.update()

        if metrics is not None:

            for name, func in metrics.items():
                tokenwise_score = evaluate_tokenwise_metric(y_true=target_seq.cpu().numpy(),
                                                            y_pred=preds,
                                                            true_lengths=true_lens,
                                                            scoring_fn=func)
                batch_scores[f"tokenwise_{name}"].append(tokenwise_score)

                examplewise_accuracy_score = evaluate_examplewise_accuracy(y_true=target_seq.cpu().numpy(),
                                                                           y_pred=preds,
                                                                           true_lengths=true_lens)
                batch_scores["example-wise_accuracy"].append(examplewise_accuracy_score)

    if batch_scores:
        epoch_scores = {name: np.mean(values) for name, values in batch_scores.items()}

    progress_bar.close()

    return losses, epoch_scores, batch_scores


def training_cycle(model,
                   train_loader,
                   validation_loader,
                   optimizer,
                   device,
                   clip,
                   metrics: Optional[Dict[str, Callable]] = None,
                   epochs: int = 1,
                   n_without_improvements: int = 5):
    train_losses = []
    validation_losses = []

    best_accuracy = 0.

    # if not os.path.exists('./models'):
    #     os.makedirs('./models')
    #
    # if not os.path.exists('./logs'):
    #     os.makedirs('./logs')

    for n_epoch in range(1, epochs + 1):
        epoch_train_losses = train(model, train_loader, optimizer, device, clip)
        epoch_validation_losses, epoch_scores, batch_scores = evaluate(model=model,
                                                                       loader=validation_loader,
                                                                       device=device,
                                                                       metrics=metrics)

        mean_train_loss = np.mean(epoch_train_losses)
        mean_validation_loss = np.mean(epoch_validation_losses)

        train_losses.append(epoch_train_losses)

        validation_losses.append(epoch_validation_losses)

        epoch_accuracy = epoch_scores["example-wise_accuracy"]

        message = "\n" + "*" * 50 + "\n"
        message += f"Epoch: {n_epoch}\n"
        message += f"Train loss: {mean_train_loss}\n"
        message += f"Validation loss: {mean_validation_loss}\n"
        message += "Validation metrics:\n"
        message += f"    examplewise_accuracy: {epoch_accuracy}\n"

        for name, score in epoch_scores.items():
            if name != "example-wise_accuracy":
                message += f"    {name}: {score}\n"

        message += "*" * 50 + "\n"
        print(message)

        if epoch_accuracy > best_accuracy:

            best_accuracy = epoch_accuracy
        #
        #     torch.save(model.state_dict(), f'models/best_language_model_state_dict.pth')
        #     torch.save(optimizer.state_dict(), 'models/best_optimizer_state_dict.pth')
        #
        else:
            break

        # torch.save(model.state_dict(), f'models/last_language_model_state_dict.pth')
        # torch.save(optimizer.state_dict(), 'models/last_optimizer_state_dict.pth')
        #
        # with open(f'logs/info_{n_epoch}.json', 'w') as file_object:
        #
        #     info = {
        #         'message': message,
        #         'train_losses': train_losses,
        #         'validation_losses': validation_losses,
        #         'train_perplexities': train_perplexities,
        #         'validation_perplexities': validation_perplexities
        #     }
        #
        #     file_object.write(json.dumps(info, indent=2))


def testing_cycle():
    pass
