# mtg-draft-agent

Authors: Matthew Smylie, Nicolas Jaramillo Torres, Mario Gomez Flores, Arpith Shanbhag, Andrei Prokhorov

Magic: The Gathering (MTG) is a popular trading card game that has been active for more than thirty years. The draft format of this game proceeds as follows: each player opens a sealed pack of cards, chooses one to keep, then passes the rest to the player next to them. This repeats until no cards remain, and then the process repeats for two other sets of packs. Drafting an optimal deck is a highly nontrivial problem, since a card’s value is dependent on the other cards already picked as well as cards that a player may expect to see later in the draft. In addition, experienced players may guess the cards that their opponents draft and strategize around that.

We examine data aggregated by fan website 17Lands, which contains records of a large number of draft games. We are primarily concerned with the overall win rates of each card within a set and the combinations of cards that are likely to appear together in winning decks. Our primary goal is to create a recommender for MTG draft, focusing on a specific card set. We will train a neural network to select, at each stage of the draft, the card that will optimize the expected win rate of the deck. We will compare its performance to that of simple algorithms based on heuristics to determine success.

Further, we seek to explore generalizations of this model to other sets by analyzing keywords in the cards’ text. We will train our model to recognize similarities in different cards’ functions. Ideally, the model will be able to predict synergies among cards even in new sets.
