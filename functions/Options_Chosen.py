# Contains options and chosen functions

# Options: returns the list of cards the player could have chosen on turn i

import pandas as pd
import numpy as np
from pathlib import Path

def options(draft_data, pack, round_num, draft_id, pack_columns):
    
    ''' Given draft_id, pack and round number, returns the avalible options. Need columns of the pack as input'''
    
    row_selection = draft_data.loc[(draft_data['pack_number'] == pack) &
                                   (draft_data['pick_number'] == round_num) &
                                   (draft_data['draft_id'] == draft_id)]
    
    if row_selection.empty:
        return []

  
    pack_data = row_selection[pack_columns]
    
    
    available_cards = pack_data.columns[pack_data.iloc[0] == 1].tolist()
    
    return available_cards


def choice(draft_data, pack, round, id):
    
    '''Given draft id, pack and round, returns the choice that was made by the player '''
   
    row = draft_data.loc[(draft_data['pack_number'] == pack) &
                         (draft_data['pick_number'] == round) &
                         (draft_data['draft_id'] == id)]

    
    return row['pick']



 





# Chosen: returns the card that was chosen by the player on turn i


