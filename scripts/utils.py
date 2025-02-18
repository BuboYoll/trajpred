import logging
import os
from datetime import datetime
from pathlib import Path

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
    logging.info(f"Logging setup complete at {tim}.")

def valid_log():
    parent = Path(__file__).parent
    log_dir = parent.parent / 'logs'

    tim = datetime.now()
    month = tim.month
    day = tim.day
    hour = tim.hour
    s = f'evaluate_{month}_{day}_{hour}'
    log_path = os.path.join(log_dir, s)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Logging setup complete at {tim}.")