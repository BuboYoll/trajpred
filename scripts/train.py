import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
from utils import train_log
from scripts.dataset import load_data
from model import TrajPreLocalAttnLong
from config import TrainConfig, DataConfig, ModelConfig
from pathlib import Path
from torch.optim.lr_scheduler import StepLR


def train(model_path=None):
    """
    param
        model_path: the model's path. If not provided, a new model would be initialized
    """
    train_loader, valid_loader = load_data()

    # init logger
    train_log()

    #hyper params
    device = TrainConfig.device
    epochs = TrainConfig.epochs
    lr = TrainConfig.lr
    target_len = DataConfig.target_len # stay consistency with the loader data.

    model = TrajPreLocalAttnLong(
        loc_size=ModelConfig.loc_size,
        loc_emb_size=ModelConfig.loc_emb_size,
        time_size=ModelConfig.tim_size,
        time_emb_size=ModelConfig.tim_emb_size,
        hidden_size=ModelConfig.hidden_size
    )
    model.to(device)

    criterion = F.nll_loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    # if model path provided, load the model
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Training from scratch...")

    running_loss = []

    for epoch in range(epochs):
        model.train()
        for i, ((_, loc, tim), (target_loc, _)) in enumerate(train_loader):
            loc = loc.to(device)
            tim = tim.to(device)
            target_loc = target_loc.to(device)

            outputs = model(loc, tim, target_len)  # B, target_len, C
            B, L, C = outputs.shape
            outputs = outputs.reshape(B * L, C)
            target_loc = target_loc.reshape(B * L, )

            loss = criterion(outputs, target_loc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record every 500 batches
            if (i+1) % 1000 == 0:
                running_loss.append(loss.item())
                logging.info(
                    f'Epoch [{epoch + 1}/{epochs}] : Batch [{i + 1}/{len(train_loader)}] || loss: {running_loss[-1]:.4f} || avg Loss: {sum(running_loss) / len(running_loss):.4f}')

        # evaluate
        model.eval()
        val_loss = 0.0
        corc = 0
        total = 0
        with torch.no_grad():
            for ((_, loc, tim), (target_loc, _)) in valid_loader:
                loc = loc.to(device)
                tim = tim.to(device)
                target_loc = target_loc.to(device)

                outputs = model(loc, tim, target_len)

                outputs = outputs.reshape(B * L, C)
                target_loc = target_loc.reshape(B * L, )

                val_loss += criterion(outputs, target_loc).item()
                _, predicted = outputs.max(dim=1)

                corc += predicted.eq(target_loc).sum().item()
                total += predicted.shape[0]

        # valid loss
        val_loss /= len(valid_loader)
        val_accuracy = 100. * corc / total
        logging.info('====')
        logging.info(
            f'Validation: Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.3f}%')
        logging.info('====')
        #record model
        # if (epoch + 1) % 5 == 0:
        #     torch.save(model.state_dict(), f'ep{epoch+1}_valacc{val_accuracy}')
        #     logging.info(f'Model saved at epoch {epoch + 1}')

    p = Path(__file__).resolve().parent.parent
    save_p = p / 'models' / 'model2.pth'
    torch.save(model.state_dict(), save_p)
    logging.info('Training finished.')

train()