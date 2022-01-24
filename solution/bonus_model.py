"""Define your architecture here."""
import torch
from torch import nn
import os
from models import get_effecient_model


def my_bonus_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    if os.path.isfile('checkpoints/bonus_model.pt'):
        pre_trained = False
    else:
        pre_trained = True
    model = get_effecient_model(pre_trained)
    if pre_trained:
        # should add option to train model
        pass
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/bonus_model.pt')['model'])
    return model

