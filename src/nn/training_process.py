import json
import os
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from src.utils.metrics import evaluate_batch


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

    for index_seq, encoder_seq, target_seq, true_lens in loader:
        index_seq = index_seq.squeeze(0)
        encoder_seq = encoder_seq.to(device).squeeze(0)
        target_seq = target_seq.to(device).squeeze(0)
        true_lens = true_lens.to(device).squeeze(0)

        loss = model(encoder_seq, target_seq, true_lens)
        # loss = loss.view(encoder_seq.size(0), encoder_seq.size(1))
        # loss = torch.mean(torch.sum(loss, axis=1))

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
             verbose: bool = True) -> Tuple[list, dict, dict]:
    losses = []
    epoch_scores = dict()
    batch_scores = defaultdict(lambda: [])

    progress_bar = tqdm(total=len(loader), disable=not verbose, desc='Evaluate')

    model.eval()

    for index_seq, encoder_seq, target_seq, true_lens in loader:
        index_seq = index_seq.squeeze(0)
        encoder_seq = encoder_seq.to(device).squeeze(0)
        target_seq = target_seq.to(device).squeeze(0)
        true_lens = true_lens.to(device).squeeze(0)

        with torch.no_grad():
            loss = model(encoder_seq, target_seq, true_lens)
            preds = model.predict(encoder_seq, true_lens)

        losses.append(loss.item())

        progress_bar.set_postfix(loss=np.mean(losses[-last_n_losses:]))

        progress_bar.update()

        if metrics is not None:

            batch_scores = evaluate_batch(y_true=target_seq,
                                          y_pred=preds,
                                          metrics=metrics,
                                          batch_scores=batch_scores,
                                          true_lengths=true_lens)

    if batch_scores:
        epoch_scores = {name: np.mean(values) for name, values in batch_scores.items()}

    progress_bar.close()

    return losses, epoch_scores, batch_scores


def training_cycle(experiment_id: str,
                   model: Module,
                   train_loader: DataLoader,
                   validation_loader: DataLoader,
                   optimizer,
                   device: torch.device,
                   clip: float,
                   metrics: Optional[Dict[str, Callable]] = None,
                   epochs: int = 1,
                   n_without_improvements: int = 2,
                   early_stopping: bool = True,
                   save_best: bool = True,
                   save_last: bool = True,
                   write_log: bool = True,
                   log_save_dir: Optional[str] = None):

    model_name = model.__class__.__name__

    train_losses = []
    validation_losses = []

    impatience = 0

    best_accuracy = 0.

    model_save_dir = f"./models/{model_name}/{experiment_id}"
    if (save_last or save_best) and os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if write_log and not log_save_dir:
        log_save_dir = f"./logs/{model_name}/{experiment_id}"
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)

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

        epoch_accuracy = epoch_scores["example_accuracy"]

        message = "\n" + "*" * 50 + "\n"
        message += f"Epoch: {n_epoch}\n"
        message += f"Train loss: {mean_train_loss:.6f}\n"
        message += f"Validation loss: {mean_validation_loss:.6f}\n"
        message += "Validation metrics:\n"
        message += f"    example_accuracy: {epoch_accuracy:.6f}\n"

        for name, score in epoch_scores.items():
            if name != "example_accuracy":
                message += f"    {name}: {score:.6f}\n"

        message += "*" * 50 + "\n"
        print(message)

        if epoch_accuracy > best_accuracy:

            print(f"Accuracy improved from the previous best of {best_accuracy:.6f}")

            best_accuracy = epoch_accuracy
            impatience = 0

            if save_best:
                print(f"Saving the current best state...")
                torch.save(model.state_dict(), f"{model_save_dir}/best_model_state_dict.pth")
                torch.save(optimizer.state_dict(), f"{model_save_dir}/best_optimizer_state_dict.pth")

        else:
            print(f"Accuracy did not improve from the previous best of {best_accuracy:.6f}")
            impatience += 1
            if early_stopping:
                print(f"Current impatience: {impatience}")
                print(f"Stopping at {n_without_improvements} impatience")

        if early_stopping and impatience > n_without_improvements:
            print(f"Early stopping because there was no improvement for {impatience} epochs")
            if write_log:
                with open(f"{log_save_dir}/best_score.jsonl", "w") as file:
                    info = {"best_accuracy": best_accuracy,
                            "n_epoch": n_epoch,
                            "stopped_early": True}
                    file.write(json.dumps(info) + "\n")
            break

        if save_last:
            print(f"Saving the current state...")
            torch.save(model.state_dict(), f"{model_save_dir}/last_model_state_dict.pth")
            torch.save(optimizer.state_dict(), f"{model_save_dir}/last_optimizer_state_dict.pth")

        if write_log:
            with open(f"{log_save_dir}/train_log.jsonl", "a") as file:
                info = {"n_epoch": n_epoch,
                        "mean_train_loss": mean_train_loss,
                        "mean_validation_loss": mean_validation_loss,
                        "impatience": impatience,
                        **epoch_scores}
                file.write(json.dumps(info) + "\n")

    if save_best:
        print("Loading the best model...")
        model.load_state_dict(torch.load(f"{model_save_dir}/best_model_state_dict.pth"))

    if write_log:
        with open(f"{log_save_dir}/best_train_score.jsonl", "w") as file:
            info = {"best_accuracy": best_accuracy,
                    "n_epoch": n_epoch,
                    "stopped_early": False}
            file.write(json.dumps(info) + "\n")
