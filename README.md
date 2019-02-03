# POS-tagging
Perform parts-of-speech tagging in Python using Bayesian Networks.

This program works on a limited set of 12 tags learned from a training file with ~1M words in ~50K sentences. It uses 3 different Bayes' Nets to predict the tags associated with words:
  1. Simplified Model (no conncection between words)
  2. Hidden Markov Chains (each word is dependent on its previous word only)
  3. Complex Model (each word is dependent on two of its previous words)

While network #1 uses Naive Bayes algorithm, #2 uses Viterbi algorithm, and #3 uses Markov Chain Monte Carlo (MCMC) to make the predictions.

The program runs as follows in the command line:
./label.py training_file testing_file
The file format of the datasets is quite simple: each line consists of a word, followed by a space, followed by one of 12 part-of-speech tags: ADJ (adjective), ADV (adverb), ADP (adposition), CONJ (conjunction), DET (determiner), NOUN, NUM (number), PRON (pronoun), PRT (particle), VERB, X (foreign word), and . (punctuation mark). Sentence boundaries are indicated by blank lines.

Additionally, the program outputs the logarithm of the posterior probability for each solution it finds under each of the three models along with a running evaluation showing the percentage of words and whole sentences that have been labeled correctly according to the ground truth so far.
