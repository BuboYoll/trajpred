class ModelConfig:
    loc_size = 61858
    tim_size = 168
    loc_emb_size = 200
    tim_emb_size = 20
    hidden_size = 40
    dropout_p = 0.2

class DataConfig:
    batch_size = 64
    top_locs = 20
    held_out = 0.9
    min_datapoints = 21
    traj_len = 21
    target_len = 1 # also used in the forward method in model

class TrainConfig:
    lr = 0.001
    epochs = 20
    device = 'cuda'



