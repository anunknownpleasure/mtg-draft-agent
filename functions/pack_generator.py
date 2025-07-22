import numpy as np

# Generate a booster pack of 14 cards from the expansion list (as a json list), where each pack has:
## 7 common cards
## 3 uncommon cards
## 1 guaranteed rare (87.5% chance) or mythic (12.5% chance)
## 2 cards of any type or rarity
## 1 common land

def generate_pack(expansion):
  mythic_cards = [card for card in expansion if card['rarity']=='mythic']
  rare_cards = [card for card in expansion if card['rarity']=='rare']
  uncommon_cards = [card for card in expansion if card['rarity']=='uncommon']
  common_cards = [card for card in expansion if card['rarity']=='common']
  land_cards = [card for card in expansion if ('Land' in card['type_line'] and card['rarity']=='common')]

  #Roll a die to determine if the rare is a mythic instead.
  d = np.random.randint(1,9)
  if d == 1:
    rare = np.random.choice(mythic_cards,1)
  else:
    rare = np.random.choice(rare_cards,1)

  uncommons = np.random.choice(uncommon_cards,3)
  commons = np.random.choice(common_cards,7)
  wilds = np.random.choice(expansion,2)
  land = np.random.choice(land_cards,1)

  pack = np.concatenate((rare, uncommons, commons, wilds, land))
  return pack
