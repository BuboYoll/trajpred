import torch
import torch.nn.functional as F
import logging
import numpy as np
from config import ModelConfig, DataConfig
from model import TrajPreLocalAttnLong
from scripts.dataset import load_data
from scripts.utils import test_log
from collections import Counter

model = TrajPreLocalAttnLong(
    loc_size=ModelConfig.loc_size,
    loc_emb_size=ModelConfig.loc_emb_size,
    time_size=ModelConfig.tim_size,
    time_emb_size=ModelConfig.tim_emb_size,
    hidden_size=ModelConfig.hidden_size
)
device = 'cuda'

model.load_state_dict(torch.load('models/model2.pth', weights_only=True))
model.to(device)
model.eval()

val_loss = 0.0
corc = 0
total = 0
target_len = DataConfig.target_len
criterion = F.nll_loss
batch_size = DataConfig.batch_size

test_log()
with torch.no_grad():
    _, valid_loader = load_data()
    check = {}
    for ((uid, loc, tim), (target_loc, _)) in valid_loader:
        uid = uid.to(device)
        loc = loc.to(device)
        tim = tim.to(device)
        target_loc = target_loc.to(device)

        outputs = model(loc, tim, target_len)

        B, L, C = outputs.shape
        outputs = outputs.reshape(B * L, C)
        _, pred = outputs.max(dim=1)  # pred: (B*L,)
        target_loc = target_loc.reshape(B * L, )

        # for each user, record the correct and wrong prediction. {user: [[predicted locations], [real locations]]}
        for i, user in enumerate(uid):
            user = user.item()
            if not(user in check.keys()):
                # user currently not in check dict
                check[user] = [[pred[i].item()], [target_loc[i].item()]]
            else:
                # user is in check dict
                rec = check[user]
                rec[0].append(pred[i].item())
                rec[1].append(target_loc[i].item())

        corc += pred.eq(target_loc).sum().item()
        total += pred.shape[0]
        val_loss /= len(valid_loader)
        val_accuracy = 100. * corc / total

# valid loss
val_loss /= len(valid_loader)
val_accuracy = 100. * corc / total
logging.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.3f}%')

for uid in check.keys():
    res = check[uid]
    a, b = np.array(res[0]), np.array(res[1])
    check[uid].append(a[a==b])
    check[uid].append(a[a!=b])

def correct_counter(check: dict):
    cor_cnt = {}  # correct location count
    wro_cnt = {}  # wrong location count
    for uid in check.keys():
        correct = check[uid][2]  # the correct prediction
        wrong = check[uid][3]
        cur_c = dict(Counter(correct))  # count the current correct number
        cur_w = dict(Counter(wrong))  # count the current wrong number
        for loc in cur_c.keys():  # locations in the counter of correct preditions
            if not(loc in cor_cnt.keys()):  # not appeared in cor_cnt
                cor_cnt[loc] = cur_c[loc]
            else:  # exist. sum up
                cor_cnt[loc] += cur_c[loc]

        for loc in cur_w.keys():
            if not(loc in wro_cnt.keys()):
                wro_cnt[loc] = cur_w[loc]
            else:
                wro_cnt[loc] = cur_w[loc]

    return cor_cnt, wro_cnt

correct, wrong = correct_counter(check)

