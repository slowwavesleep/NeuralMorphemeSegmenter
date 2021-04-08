import numpy as np
from tqdm import tqdm
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import json
import os

from src.utils.metrics import evaluate_metric_padded


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
             last_n_losses: int = 500,
             verbose: bool = True):
    losses = []

    progress_bar = tqdm(total=len(loader), disable=not verbose, desc='Evaluate')

    model.eval()

    for encoder_seq, target_seq, true_lens in loader:
        encoder_seq = encoder_seq.to(device)
        target_seq = target_seq.to(device)

        with torch.no_grad():
            loss = model(encoder_seq, target_seq)
            preds = model.predict(encoder_seq)

        losses.append(loss.item())

        score = evaluate_metric_padded(target_seq.cpu().numpy(), preds, true_lens)

        progress_bar.set_postfix(loss=np.mean(losses[-last_n_losses:]),
                                 score=score)

        progress_bar.update()

        # print(model.predict(encoder_seq))

    progress_bar.close()

    return losses


def training_cycle(model,
                   train_loader,
                   validation_loader,
                   optimizer,
                   device,
                   clip,
                   epochs: int = 1):
    train_losses = []
    validation_losses = []

    best_validation_loss = 1e+6

    # if not os.path.exists('./models'):
    #     os.makedirs('./models')
    #
    # if not os.path.exists('./logs'):
    #     os.makedirs('./logs')

    for n_epoch in range(1, epochs + 1):

        epoch_train_losses = train(model, train_loader, optimizer, device, clip)
        epoch_validation_losses = evaluate(model, validation_loader, device)

        mean_train_loss = np.mean(epoch_train_losses)
        mean_validation_loss = np.mean(epoch_validation_losses)

        train_losses.append(epoch_train_losses)

        validation_losses.append(epoch_validation_losses)

        # message = f'Epoch: {n_epoch}\n'
        #
        # print(message)

        # if mean_validation_loss < best_validation_loss:
        #
        #     best_validation_loss = mean_validation_loss
        #
        #     torch.save(model.state_dict(), f'models/best_language_model_state_dict.pth')
        #     torch.save(optimizer.state_dict(), 'models/best_optimizer_state_dict.pth')
        #
        # else:
        #     break

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
