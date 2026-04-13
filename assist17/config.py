import torch
import random

class Config:
    device = torch.device("cuda")
    MAX_SEQ = 1000
    EMBED_DIMS = 512
    ENC_HEADS = DEC_HEADS = 8
    NUM_ENCODER = NUM_DECODER = 4
    BATCH_SIZE = 128
    TRAIN_FILE = "firstclassstudent171.csv"
    TOTAL_EXE = 2852
    TOTAL_SKI = 329
    HARD = 10000
    CORRECT = 10000
 
