import numpy as np
import pandas as pd

from itertools import product
from time import time

###############
# Functions to neatly print card attributes
###############
# Checks if a card has multiple sides
def is_two_sided(card):
    if "card_faces" in card.keys() and card["card_faces"] is not None:
        return True
    else:
        return False


# Print attributes of a single card.
# Attributes depend on the type of card:
# - Lands do not have mana cost
# - Only creatures have power and toughness
def print_single_card(card):
    card_type = card["type_line"]

    print(f"Name: {card['name']}")

    # Show rarity if available
    if "rarity" in card.keys():
        print(f"Rarity: {card['rarity']}")

    # Everything but lands has a mana cost
    if card_type != "Land":
        print(f"Cost: {card['mana_cost']}")

    print(f"Type: {card['type_line']}")

    # Only creatures have power and toughness
    if "Creature" in card_type:
        print("P/T: {}/{}".format(card["power"], card["toughness"]))

    # Only Planeswalkers have power and toughness
    if "Planeswalker" in card_type:
        print("L: {}".format(card["loyalty"]))

    print()
    print(card["oracle_text"])


# Print attributes of a card
# This function checks if a card has two sides, in which case it prints
# both sides separately
def print_card_attributes(card):
    print("--------------- ")

    if is_two_sided(card):
        faces = card["card_faces"]

        # Print full name and rarity only once
        print(card["name"])
        print(f"Rarity: {card['rarity']}")
        for i, face in enumerate(faces):
            print("// --- //")
            print_single_card(face)

            if i < len(faces) - 1:
                print()

    else:
        print_single_card(card)

    print("--------------- ")
    print()


###############
# Reading draft history
###############
def get_cards_from_draft_df(draftdata, prefix="pack_card_"):
    """
    Gets the list of cards in an expansion by looking at the column names
    of draftdata. draftdata has a one-hot encoding of the cards seen and
    picked in a draft, and it stores them in columns named
    {prefix}_{card_name}. We remove that prefix and make a list of the card
    names.

    Args:
    - draftdata: A Dataframe of played drafts from 17Lands
    - prefix: If the format were to change, you can choose the new prefix
              here

    Returns:
    - card_names: List of card names
    - card_to_idx, idx_to_cards: Dictionaries that translate between cards
                                 and their index in the card_names list.
    """
    # The card names appear in the columns that start with a fixed prefix
    card_names = draftdata.filter(regex=prefix).columns
    card_names = list(card_names)

    # Remove prefix
    card_names = [name.replace(prefix, "") for name in card_names]

    # Dictionaries to translate between cards and indices
    card_to_idx = {}
    idx_to_card = {}

    for idx, card in enumerate(card_names):
        card_to_idx[card] = idx
        idx_to_card[idx] = card

    return card_names, card_to_idx, idx_to_card


def count_to_list(row, prefix):
    """
    Turns a vector of counts into a list of card names, each one
    repeated as many times as the vector's entry. We obtain the
    name of the cards by extracting the names of the columns with
    non-zero value and removing the given column prefix from it.

    Args:
    - row: A row of a 17Lands dataframe containing draft histories.
           The row should correspond to specific pick and pack
           numbers.
    - prefix: Either 'pack_card_' to count the cards seen in each
              round of the draft, or 'pool_' to count the cards in
              the player's pool (i.e. cards chosen in previous
              round).
    """
    # Filter only columns with the input prefix and tranpose
    df_prefix = row.filter(regex=prefix)
    df_prefix = df_prefix.transpose()

    # Get rows whose entry is not 0
    idx_orig = row.index[0]
    column_list = df_prefix[df_prefix[idx_orig] > 0].index

    # Remove prefix and add repetitions
    card_list = []
    for col_name in column_list:
        card_name = col_name.replace(prefix, "")
        repetitions = row.loc[idx_orig, col_name]

        card_list.extend([card_name] * repetitions)

    return card_list


def get_played_drafts(
    draftdata,
    card_to_idx=None,
    num_packs=3,
    num_picks=14,
    prefix_pack="pack_card_",
    prefix_pool="pool_",
):
    """
    Creates a dataframe with played drafts. Cards are tokenized with card_to_idx.

    Args:
    - draftdata: A Dataframe of played drafts from 17Lands
    - card_to_idx (optional): A card-index dictionary, as produced
                              by get_cards_from_draft_df. If not given,
                              we build it ourselves with get_cards_from_draft_df.
    - num_packs, num_picks: Number of packs and cards per pack. May change
                            given the draft format.
    - prefix_pack, prefix_pool: Prefixes of the column names in the 17Lands
                                dataframes that contain cards in pack and
                                cards in pool.

    Returns:
    - drafts: A dataframe that, for each draft_id and for each turn of the
      draft, it has the following columns:
            - pick: Card chosen in a given turn
            - pack: Cards in pack in a given turn i
            - pool: List of cards chosen up to turn i-1
    """
    # Get unique ids
    draft_ids = draftdata["draft_id"].unique()

    # Get columns with the player's options
    pack_columns = draftdata.filter(regex=prefix_pack).columns
    pack_columns = list(pack_columns)

    # Get columns with the player's pool of cards
    pool_columns = draftdata.filter(regex=prefix_pool).columns
    pool_columns = list(pool_columns)

    # Get the card-index dictionary if we don't already have it
    if card_to_idx is None:
        card_names, card_to_idx, idx_to_card = get_cards_from_draft_df(draftdata)

    # Compile data for each draft_id
    draft_list = []

    for i, id in enumerate(draft_ids):
        time_start = time()

        # Get draft info for id
        data_id = draftdata.loc[draftdata["draft_id"] == id, :]

        # Check that we have the right amount of data
        num_rows = data_id.shape[0]
        if num_rows != num_packs * num_picks:
            print(f"{i+1}/{len(draft_ids)}", end=": ")
            print(
                f"Draft incomplete. Only {num_rows} out of {num_packs*num_picks} rows. Skipping id {id}."
            )

            continue

        # Build iterators to extract information in turn order
        draft_turns = product(range(num_packs), range(num_picks))

        for pack_idx, pick_idx in draft_turns:
            # Get row for the turn by filtering pack number, pick number, and draft id
            df_turn = draftdata[
                (draftdata["draft_id"] == id)
                & (draftdata["pack_number"] == pack_idx)
                & (draftdata["pick_number"] == pick_idx)
            ]

            # Get pick, cards in pack, and cards in pool
            df_index = df_turn.index[0]
            pick = df_turn.at[df_index, "pick"]
            cards_in_pack = count_to_list(df_turn, prefix_pack)
            cards_in_pool = count_to_list(df_turn, prefix_pool)

            # Store results
            chosen = card_to_idx[pick]
            pack = [card_to_idx[card] for card in cards_in_pack]
            pool = [card_to_idx[card] for card in cards_in_pool]

            row = {
                "draft_id": id,
                "pack_number": pack_idx,
                "pick_number": pick_idx,
                "pick": chosen,
                "pack": pack,
                "pool": pool,
            }
            draft_list.append(row)

        time_end = time()
        dt = time_end - time_start
        print(f"{i+1}/{len(draft_ids)}: {np.round(dt,3)}")
    print()

    # Convert draft_list into DataFrame
    time_start = time()
    drafts = pd.DataFrame(
        draft_list,
        columns=["draft_id", "pack_number", "pick_number", "pick", "pack", "pool"],
    )
    time_end = time()
    print(f"End: {np.round(dt,3)}")

    return drafts
