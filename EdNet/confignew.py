import torch
import random

class Config:
    device = torch.device("cuda")
    MAX_SEQ = 100
    EMBED_DIMS = 512
    ENC_HEADS = DEC_HEADS = 8
    NUM_ENCODER = NUM_DECODER = 4
    BATCH_SIZE = 32
    TRAIN_FILE = "ednetnewdatalayer1.csv"
    TOTAL_EXE = 13523
    TOTAL_SKI = 188
    HARD = 100
    CORRECT = 100
 
