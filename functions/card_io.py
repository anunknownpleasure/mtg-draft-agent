# API requests
import requests

# Utilities
import os
import pathlib
from time import time, sleep
import json

# Warnings
import warnings


###############
# Functions to neatly print card attributes
###############
# Checks if a card has multiple sides
def is_two_sided(card):
  if 'card_faces' in card.keys() and card['card_faces'] is not None:
    return True
  else:
    return False

# Print attributes of a single card.
# Attributes depend on the type of card:
# - Lands do not have mana cost
# - Only creatures have power and toughness
def print_single_card(card):
  card_type = card['type_line']

  print(f"Name: {card['name']}")

  # Show rarity if available
  if 'rarity' in card.keys():
    print(f"Rarity: {card['rarity']}")

  # Everything but lands has a mana cost
  if card_type != "Land":
    print(f"Cost: {card['mana_cost']}")

  print(f"Type: {card['type_line']}")

  # Only creatures have power and toughness
  if 'Creature' in card_type:
    print('P/T: {}/{}'.format(card['power'], card['toughness']))

  # Only Planeswalkers have power and toughness
  if 'Planeswalker' in card_type:
    print('L: {}'.format(card['loyalty']))

  print()
  print(card['oracle_text'])

# Print attributes of a card
# This function checks if a card has two sides, in which case it prints
# both sides separately
def print_card_attributes(card):
  print('--------------- ')

  if is_two_sided(card):
    faces = card['card_faces']

    # Print full name and rarity only once
    print(card['name'])
    print(f"Rarity: {card['rarity']}")
    for i,face in enumerate(faces):
      print('// --- //')
      print_single_card(face)

      if i<len(faces)-1:
        print()

  else:
    print_single_card(card)

  print('--------------- ')
  print()


###############
# Scraping Scryfall
###############
headers_ = {
    'User-Agent': '2025-dl-mtg-draft',  # Name of our app
    'Accept': 'application/json'        # Return format in json
}
scryfall_url_ = "https://api.scryfall.com/cards/search"

def get_expansion_from_scryfall(expansion, delay=50e-3, return_json=False):
    """
    Get all cards from a specific expansion.
    expansion: Expansion code (e.g., 'm21' for Magic 2021)
    return: List of cards in the expansion

    Note: This function handles pagination and rate limits.
    """
    if delay < 50e-3:
      warnings.warn(f'delay ({delay}) is less than 50 milliseconds. Scryfall recommends a delay of 50-100 milliseconds between requests to avoid rate limiting.')
    
    # Response has several pages, we loop over them
    has_more = True
    page_num = 1
  
    responses = []        # Full response from the API (if needed)
    responses_json = []   # Contains only the json files
    cards = []            # Contains only the cards
    while has_more:
        params = {
            'q': f'set:{expansion}',
            'page': str(page_num)
        }
        response = requests.get(scryfall_url_, headers=headers_, params=params)

        # Scryfall asks us to add a 50-100 millisecond delay between requests
        sleep(delay)

        # Store full responses if we need them
        if return_json:
          responses.append(response)
          responses_json.append(response.json())

        # Store the cards
        cards.extend(response.json()['data'])

        # Check if there are more pages
        has_more = response.json()['has_more']
        page_num += 1

    if return_json:
      return cards, responses, responses_json
    else:
      return cards
  
def get_saved_expansion(expansion):
  """
  Get the cards from a saved expansion file.
  expansion: Expansion code (e.g., 'm21' for Magic 2021)
  return: List of cards in the expansion
  """
  folder = 'MTGdraft/Scryfall-data/card-sets'
  filename = f"expansions/{expansion}.json"
  path = pathlib.Path(folder, filename)

  with open(path, 'r') as file:
      cards = json.load(file)

  return cards