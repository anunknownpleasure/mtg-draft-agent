import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from datasets import Dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Use the number of cards in the DSK expansion.
PAD_IDX = 286
max_pack_size = 14

class DraftBotLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers=1,
        padding_idx=None,
        p_LSTM=0.0,
        p_out=0.0,
    ):
        super().__init__()

        # Attributes
        self.vocab_size = vocab_size

        # For padding sequences. We later add a "word" to the embedding layer to
        # encode this new padding index
        if padding_idx is None:
            self.padding_idx = vocab_size
        else:
            self.padding_idx = padding_idx

        # LSTM followed by a full layer that produces logits
        self.embedding = nn.Embedding(
            vocab_size + 1, embed_dim, padding_idx=self.padding_idx
        )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=p_LSTM,
        )
        self.dropout = nn.Dropout(p=p_out)
        self.output_layer = nn.Linear(
            hidden_dim, vocab_size
        )  # maps hidden state to logits

    # Returns the device the model sits on
    def device(self):
        return next(self.parameters()).device

    def forward(self, pack_batch, hidden_state=None):
        """
        pack_batch: [Tensor] of shape (batch_size, pack_size). Contains a batch of packs.
        hidden_state: Tuple of (h_0, c_0), each of shape (batch_size, 1, hidden_dim)
        Returns:
            logits: shape (batch_size, seq_len, pack_size) with seq_len=1
            hidden_state: final hidden state (for next step)
        """
        pack_embed = self.embedding(pack_batch)  # (batch_size, pack_size, embed_dim)
        # print(' ------------ ')
        # print('Embedding:', pack_embed.shape)

        # Careful with padded tokens
        mask = (pack_batch != self.padding_idx).unsqueeze(
            -1
        )  # (batch_size, seq_len, 1)
        valid_counts = mask.sum(dim=1, keepdim=True)  # (batch_size, 1, 1)

        # pool cards into a single vector (i.e. change pack_size to 1)
        pack_pooled = pack_embed.sum(
            dim=1, keepdim=True
        )  # shape (batch_size, 1, embed_dim)
        pack_pooled = pack_pooled / valid_counts

        # print('Embedding pooled:', pack_pooled.shape)
        # Q: Do I need to average the embeddings? Isn't this lossy?

        # Pass through LSTM
        lstm_out, hidden = self.lstm(
            pack_pooled, hidden_state
        )  # shape: (batch_size, 1, hidden_dim)

        # print()
        # print('LSTM output')
        # print('lstm_out:', lstm_out.shape)
        # print('hidden state:', len(hidden))
        # for i,h_i in enumerate(hidden):
        #       print(f'hidden_{i}: {h_i.shape}')
        # print()

        # Dropout layer
        lstm_out_drop = self.dropout(lstm_out)

        # Map LSTM output to logits
        logits = self.output_layer(lstm_out_drop)  # shape: (batch_size, 1, vocab_size)
        # print('Logits:', logits.shape)
        # print()

        # Mask logits of tokens not in pack_batch
        mask = utils.create_logit_mask(
            pack_batch, self.vocab_size, padding_idx=self.padding_idx
        )

        # Add a dimension to match logits.shape
        mask = mask.unsqueeze(1)

        # print('mask:', mask.shape)

        # Be careful not to change tensors involved in gradient computations in place.
        # (it breaks gradient and backpropagation, possibly silently).
        # "Changing in place" means overwritting the underlying data storage.
        # masked_fill creates a new tensor, and the assignment logits = ... just changes
        # references, i.e. `logits` now points to the output of masked_fill instead of
        # the old value of logits
        logits = logits.masked_fill(~mask, -torch.inf)

        # print('New logits:', logits.shape)

        return logits, hidden

    def predict(self, pack_batch, hidden_state=None):
        """
        pack_batch: [Tensor] of shape (batch_size, pack_size). Contains a batch of packs.
        hidden_state: Tuple of (h_0, c_0), each of shape (batch_size, 1, hidden_dim)
        Returns:
            predictions: shape (batch_size,) The chosen card token from each batch
            hidden_state: final hidden state
        """
        # pack.shape = (batch_size, pack_size)
        logits, hidden = self(
            pack_batch, hidden_state
        )  # shape: (batch_size, 1, vocab_size)

        # Get predictions from the last time step
        last_logits = logits[:, -1, :]  # (batch_size, vocab_size)
        predictions = torch.argmax(last_logits, dim=-1)  # (batch,)

        return predictions, hidden

    def predict_single(self, pack, hidden_state=None):
        """
        pack: List of card tokens. We turn the pack into a tensor of shape (batch_size, pack_size)
              with batch_size=1.
        hidden_state: Tuple of (h_0, c_0), each of shape (batch_size, 1, hidden_dim)
        Returns:
            prediction: The chosen card token from the pack.
            hidden_state: final hidden state
        """
        pack_batch, _ = tensorize_lists(
            pack, device=self.device(), padding_value=self.padding_idx
        )

        return self.predict(pack_batch, hidden_state)

    def predict_batch(self, pack_list, hidden_state=None):
        """
        pack_list: List of card packs. Each pack is a list of card tokens.
                  We turn the pack list into a tensor of shape (batch_size, pack_size)
                  with batch_size=len(pack_list), and we pad all packs to the maximum
                  pack size.
        hidden_state: Tuple of (h_0, c_0), each of shape (batch_size, 1, hidden_dim)
        Returns:
            predictions: The chosen card token from each pack.
            hidden_state: final hidden state
        """
        pack_batch, _ = tensorize_lists(
            *pack_list, device=self.device(), padding_value=self.padding_idx
        )
        # pack_padded = pack_padded_sequence(pack_batch, lengths, batch_first=True, enforce_sorted=False)

        # Obtain predictions
        predictions, hidden = self.predict(pack_batch, hidden_state)
        # Note: No need to pad_packed_sequence because I return a single token

        return predictions, hidden

def tensorize_lists(*args, padding_value=PAD_IDX, device=None):
    """
    Turns the list of arguments into a padded tensor.
    - Each argument is a list of tokens, and they will be padded to have the same
      length.
    - Returns the padded tensor and the original length of each argument
    """
    # Change arguments to tensors if they aren't already
    pack_tensors = []
    for pack in args:
        # Turn list into a tensor
        if not isinstance(pack, torch.Tensor):
            # If device doesn't matter
            if device is None:
                pack_tensors.append(torch.tensor(pack))
            # Create on specified device
            else:
                pack_tensors.append(torch.tensor(pack, device=device))
        # Store the tensors
        else:
            # Device doesn't matter
            if device is None:
                pack_tensors.append(pack)
            # Move to specified device
            else:
                pack_tensors.append(pack.to(device))

    # Store lengths
    lengths = [pack.shape[0] for pack in pack_tensors]
    pack_size = max(lengths)

    # Pad to shape (batch_size, pack_size)
    pack_batch = pad_sequence(
        pack_tensors, batch_first=True, padding_value=padding_value
    )

    return pack_batch, lengths

# Dataset to return a single player's game
class PlayerDataset(Dataset):
    """
    A Dataset class that handles the players in our dataset. When called, it
    returns all the turns of a chosen player.
    """

    def __init__(self, df):
        self.df = df
        self.players = df["draft_id"].unique()
        self.padding_idx = PAD_IDX

    def __len__(self):
        """
        Number of players in our dataset
        """
        return len(self.players)

    def __getitem__(self, idx):
        """
        Returns the lists of packs and picks of the player with index idx.
        Packs and picks are sorted by turn order.
        """
        id = self.players[idx]
        df_player = self.df[self.df["draft_id"] == id]

        # Make sure that turns are sorted correctly
        df_player = df_player.sort_values(
            by=["pack_number", "pick_number"], ascending=True
        )

        # Extract packs and picks in turn sequence
        packs = []
        picks = []
        for idx, row in df_player.iterrows():
            # LSTMs (and tokens in general) work best with long dtype
            pack = torch.tensor(row["pack"], dtype=torch.long)
            pick = torch.tensor(row["pick"], dtype=torch.long)

            packs.append(pack)
            picks.append(pick)

        # Return the lists of packs and picks of a single player
        return packs, picks

    def __repr__(self):
        return (
            f"PlayerDataset with {len(self)} players\n{self.df.to_string(max_rows=5)}"
        )

    # Convenience method for counting the number of turns
    # Remember that __get_item__ returns two lists whose lengths equal
    # the number of turns
    def num_turns(self):
        return len(self.__getitem__(0)[0])


# Custom collate function.
# It's used for training the LSTM batching by player
def collate_player_turns(batch):
    """
    Collate function for PlayerDataset. Creates batches of players, and returns
    a list of batches. Each batch contains the game state (packs and picks) in a
    single turn, and for all the player in the batch. The list is sorted
    chronologically by turn.
    We assume all players have the same number of turns, and the same number of
    options in each turn.

    args:
    batch: list of tuples (packs, picks). Each tuple contains the draft data
    of a single player in turn order.

    Returns:
    pack_batches: list of packs for each turn. Each element is a tensor of shape
                  (batch_size, pack_size), where batch_size is the number of players.
    pick_batches: list of picks for each turn. Each element is a tensor of shape
                  (batch_size,), where batch_size is the number of players.
    """
    # batch: list of tuples (packs, picks). length = number of players

    # batch[0] = (packs, picks) of first player
    # num. turns = len(packs), equiv. len(picks)
    n_turns = len(batch[0][0])

    # pack and pick batches, sorted by turn
    # Element idx will be the play info of all players in turn idx
    pack_batches = []
    pick_batches = []
    for turn in range(n_turns):
        # Extract turn info from all players
        packs_turn = []
        picks_turn = []
        for player_pack, player_pick in batch:
            packs_turn.append(player_pack[turn])
            picks_turn.append(player_pick[turn])

        # Stack and store
        # We're assuming all players have the same number of options in each turn,
        # so we can stack their packs without padding.
        pack_batches.append(torch.stack(packs_turn))
        pick_batches.append(torch.stack(picks_turn))

    return pack_batches, pick_batches

# Training and evaluation functions
def train_epoch(
    model, dataloader, optimizer, loss_fn, chunk_size=max_pack_size, device=None
):
    """
    chunk_size: Back propagate over a smaller number of turns.
                The default is the size of a pack (i.e. the length of one "round" of
                drafting).
                If None, we backpropagate over all turns.
    """
    # Move model to new device
    if device is not None:
        model = model.to(device)

    # Initialize training mode
    model.train()

    # Remember: In PlayerDataset, each entry has the game information of a player,
    # which consists of two lists of length equal to the number of turns
    num_batches = len(dataloader)
    num_players = len(dataloader.dataset)
    num_turns = dataloader.dataset.num_turns()

    # Accumulate the correct picks made by all players of each batch and on each turn
    all_correct = torch.zeros(num_batches, num_turns)

    # Accumulate loss over all players and all turns
    total_loss = 0

    # Each (pack_batches, pick_batches) is a list of turn states for a player batch
    batch_count = 0
    for pack_batches, pick_batches in tqdm(dataloader):
        # Each batch is a group of players, but the last batch may be smaller
        batch_size = len(pack_batches[0])

        # Initialize variables at the start of the game
        batch_loss = 0
        hidden_state = None
        optimizer.zero_grad()

        # In case we want to backpropagate the whole game
        if chunk_size is None:
            chunk_size = num_turns

        # Play game and backpropagate every chunk_size turns
        for t0 in range(0, num_turns, chunk_size):
            # End the chunk at the game's end, not later
            chunk_end = min(t0 + chunk_size, num_turns)

            # Play chunk_size turns
            for t in range(t0, chunk_end):
                pack_batch = pack_batches[t]
                pick_batch = pick_batches[t]

                if device is not None:
                    pack_batch = pack_batch.to(device)
                    pick_batch = pick_batch.to(device)

                # Cards available to pick
                pack_size = torch.tensor(pack_batch.shape[1], device=device)

                # Forward pass -- remember hidden state from previous turn
                logits, hidden_state = model(pack_batch, hidden_state=hidden_state)

                # Note: logits is shaped (batch_size, seq_len, vocab_size) with seq_len=1
                # but loss functions such as cross entropy expect shape
                # (batch_size, vocab_size). That's why I slice here
                logits = logits[:, -1, :]

                # Accumulate losses of all players, normalized by pack size
                # if pack_size > 1:
                #     batch_loss += loss_fn(logits, pick_batch) / torch.log(pack_size)
                # else:
                #     batch_loss += loss_fn(logits, pick_batch)
                batch_loss += loss_fn(logits, pick_batch)

                # Count the number of players that picked the correct card
                predictions = torch.argmax(logits, dim=-1)  # (batch,)
                all_correct[batch_count, t] = (predictions == pick_batch).sum()

            # I accumulated losses for several players across several turns.
            # To keep magnitudes and variables interpretable (e.g. gradients),
            # I backpropagate the average loss
            played_turns = chunk_end - t0
            mean_batch_loss = batch_loss / (batch_size)

            # Backpropagate
            mean_batch_loss.backward()
            optimizer.step()

            # Reset optimizer
            optimizer.zero_grad()

            # Detach hidden state to truncate gradients every chunk_size turns
            hidden_state = tuple(h.detach() for h in hidden_state)

            # Accumulate losses of all players
            total_loss += batch_loss.item()

        # Advance batch counter
        batch_count += 1

    # Add correct choices over all batches (i.e. over all players)
    # then average over the number of players
    # accuracy_per_turn = all_correct
    accuracy_per_turn = all_correct.sum(dim=0) / num_players  # (num_turns,)

    # Average total loss over the number of players and the number of turns
    mean_loss = total_loss / (num_players)

    return mean_loss, accuracy_per_turn


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device=None):
    # Move model to new device
    if device is not None:
        model = model.to(device)

    # Initialize evaluation mode
    model.eval()

    # Remember: In PlayerDataset, each entry has the game information of a player,
    # which consists of two lists of length equal to the number of turns
    num_batches = len(dataloader)
    num_players = len(dataloader.dataset)
    num_turns = dataloader.dataset.num_turns()

    # Accumulate the correct picks made by all players of each batch and on each turn
    all_correct = torch.zeros(num_batches, num_turns)

    # Accumulate loss over all players and all turns
    total_loss = 0

    # Each (pack_batches, pick_batches) is a list of turn states for a player batch
    batch_count = 0
    for pack_batches, pick_batches in dataloader:
        # Each batch is a group of players, but the last batch may be smaller
        batch_size = len(pack_batches[0])

        # Initialize variables at the start of the game
        batch_loss = 0
        hidden_state = None

        for t in range(num_turns):
            # Extract turn info and move it to new device (if required)
            pack_batch = pack_batches[t]
            pick_batch = pick_batches[t]

            if device is not None:
                pack_batch = pack_batch.to(device)
                pick_batch = pick_batch.to(device)

            # Cards available to pick
            pack_size = torch.tensor(pack_batch.shape[1], device=device)

            # Forward pass
            logits, hidden_state = model(pack_batch, hidden_state=hidden_state)

            # Note: logits is shaped (batch_size, seq_len, vocab_size) with seq_len=1
            # but loss functions such as cross entropy expect shape
            # (batch_size, vocab_size). That's why I slice here
            logits = logits[:, -1, :]

            # Accumulate losses of all players, normalized by pack size
            # if pack_size > 1:
            #     batch_loss += loss_fn(logits, pick_batch) / torch.log(pack_size)
            # else:
            #     batch_loss += loss_fn(logits, pick_batch)
            batch_loss += loss_fn(logits, pick_batch)

            # Count the number of players that picked the correct card
            predictions = torch.argmax(logits, dim=-1)  # (batch,)
            all_correct[batch_count, t] = (predictions == pick_batch).sum()

        # Accumulate batch losses. In total, this accumulates losses of all players
        total_loss += batch_loss.item()

        # Advance batch counter
        batch_count += 1

    # Add correct choices over all batches (i.e. over all players)
    # then average over the number of players
    # accuracy_per_turn = all_correct
    accuracy_per_turn = all_correct.sum(dim=0) / num_players  # (num_turns,)

    # Average total loss over the number of players and the number of turns
    mean_loss = total_loss / (num_players)

    return mean_loss, accuracy_per_turn
