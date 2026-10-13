# How it works

This project turns a pile of chat transcripts into a small labeled dataset, trains a simple model on it, and asks one question: can the model choose which examples a human should label next, and does that beat picking them at random?

## 1. The data

The transcripts come from Anthropic's HH-RLHF dataset (helpful-base part, MIT license). Each one is a short conversation between a person and an AI assistant. The loader picks a random sample, throws away anything that matches a list of sensitive words, and stores each conversation as a document in MongoDB.

## 2. Labeling

The labeling tool shows you the conversation and the assistant's final reply. You give it a score from 1 (useless) to 5 (excellent) and write one sentence explaining why. Your answer is saved as an annotation document linked to the conversation. If you change your mind later, the old answer is kept in a history list instead of being overwritten.

## 3. The second annotator

For about 40 of the items you already rated, a second person rates them without seeing your scores. Cohen's kappa then measures how often you agree, beyond what luck would give. A value near 1 means strong agreement, near 0 means no better than chance.

## 4. The model

The model is deliberately simple. TF-IDF turns each conversation into numbers based on which words it uses, and logistic regression learns which words go with "helpful" (rating 4 or 5) versus not. It is a baseline, not a state of the art system.

## 5. Active learning

Labels are expensive, so which ones should you collect first? We compare two strategies on the same data:

- **Random:** label items in a random order.
- **Uncertainty:** after each round, ask the model which unlabeled items it is least sure about and label those next.

Because we already have labels for everything in the pool, we can replay both strategies: "labeling" an item just reveals the label we already have. Both strategies are tested on the same held-out set, repeated 20 times with different random splits, and the average curves are plotted.

## 6. Why MongoDB

Each item can carry several annotators, rationales, revision history and model outputs, and not every item has the same fields. Documents fit that better than fixed table columns, and the aggregation pipeline answers questions like rating distributions and disagreements directly.
