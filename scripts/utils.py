import logging
import os
from datetime import datetime
from pathlib import Path
from config import DataConfig, ModelConfig, TrainConfig

def train_log():
    parent = Path(__file__).parent
    log_dir = parent.parent / 'logs'

    tim = datetime.now()
    month = tim.month
    day = tim.day
    hour = tim.hour
    s = f'train_{month}_{day}_{hour}'
    log_path = os.path.join(log_dir, s)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"DataConfig: batchsize: {DataConfig.batch_size} target_len: {DataConfig.target_len} traj_len: {DataConfig.traj_len}\n "
                 f"TrainConfig: lr: {TrainConfig.lr}, epochs: {TrainConfig.epochs}\n"
                 f"ModelConfig: loc_emb: {ModelConfig.loc_emb_size}, tim_emb: {ModelConfig.tim_emb_size}, hidden: {ModelConfig.hidden_size}, dropout: {ModelConfig.dropout_p}")

def test_log():
    parent = Path(__file__).parent
    log_dir = parent.parent / 'logs'

    tim = datetime.now()
    month = tim.month
    day = tim.day
    hour = tim.hour
    s = f'test_{month}_{day}_{hour}'
    log_path = os.path.join(log_dir, s)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Logging setup complete at {tim}.")