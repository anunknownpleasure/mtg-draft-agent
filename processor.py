import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# from datasets import Dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import pathlib
from itertools import product
from sklearn.model_selection import train_test_split

from time import time
from tqdm.auto import tqdm

import functions.card_io as card_io
import functions.utils as utils

from pyarrow.parquet import ParquetFile
import pyarrow as pa

n = 100000
data_file = 'rawdata/DSK_PremierDraft_draft.parquet'
output_file = 'clean_data/DSK_drafts.parquet'

pf = ParquetFile(data_file)
first_n_rows = next(pf.iter_batches(batch_size = n))
draftdata = pa.Table.from_batches([first_n_rows]).to_pandas()

# Get unique draft ids
draft_ids = draftdata["draft_id"].unique()

# Get card names and card-index dictionaries
card_names, card_to_idx, idx_to_card = card_io.get_cards_from_draft_df(draftdata)
vocab_size = len(card_names)

# Get draft history
drafts = card_io.get_played_drafts(draftdata, card_to_idx)
max_pack_size = drafts["pick_number"].max() + 1

# Use expansion size as padding_idx
PAD_IDX = vocab_size

drafts.to_parquet(output_file)
