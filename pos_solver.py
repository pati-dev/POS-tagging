###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids: Ankit Mathur (anmath), Nitesh Jaswal (njaswal), and Nishant Jain (nishjain)
#
# (Based on skeleton code by D. Crandall)
#
####
# Report:
## Training phase:
# We created a hash to store P(initial_state) and hash of hashes to store P(transition)
# and P(emission).
# First, we tried to update these 3 hashes while reading the train data, but we observed
# that it took ~7.5 minutes for training the model. Reason, as we found out after
# several hours of debugging, was that as the size of the hash tables increased,
# it took longer and longer for the machine to update the frequencies in the hashes.
# Hence, as a logical next step, we stored the frequencies for each sentence separately
# in empty hashes that we planned to merge at a later stage once the entire training data
# was read.
# For merging these local hashes to create the global hashes, we tried using the collections.Counter function
# that can simply add up the frequencies for already existing keys between local and global hashes,
# and accounts for unavailable keys by adding them to the global hash from the local hash.
# However, this did not help in reducing the time at all and we were back at ~7 minutes for training.
# Finally, we got rid of the collections.Counter and merged the local hashes into global hashes by
# looping over each local hash pertaining to each sentence. This finally did the trick and we
# were able to do the training in ~3 seconds - down from 450 seconds! HUGE WIN!!
# Please note that we have passed the probabilities through the negative log function (base 10) to ensure that
# we do not run into floating point errors. Hence, we always add them with each other, which is equivalent to
# multiplying the probabilities with each other. This also means that a variable with higher sum has a lower probability.
## Testing phase:
# As described above, in the testing phase we aim to choose tags with lower probability. Hence, we have chosen to
# initialize the probabilities as infinity.
# In order to deal with probabilities involving words or tags that were not learned by the model during training,
# we have, for no particular reason, chosen to assume (the negative log10 of) their probability to be 1 million.
## Simplified model:
# Testing as per the Simplified model was pretty straight-forward. Please note that for cases where the word was
# not found in any of the emission probabilities, i.e. P( word | tag ), we have assumed it to be a noun, as was intuitive to us.
# Results as per Simplified model:
# Words correct: ~92.91%       Sentences correct: ~43.15%
## HMM using Viterbi:
# Testing as per HMM using Viterbi decoding was the motivation behind modifying the probabilities to negative logarithms.
# We created 2 hashes: one to store the probability values in the viterbi table and
# another one to store the sequence, which is to identify the value in the previous
# column of the viterbi table that was chosen to as the maximum for current column.
# As with the the Simplified model, we initialize the initial probability to be infinity.
# Results as per HMM:
# Words correct: ~95.06%       Sentences correct: ~54.45%
## Complex model using MCMC:
# Since this model requires transition probabilities from the tag two words before to the current word,
# we revisited the training phase and added an additional var to store these transition probabilities.
# In order to implement the "flipping of a biased coin" when sampling from the posterior of the complex
# model, we used the numpy.random.choice function. Since this function requires an array of probabilities
# to be passed into it, we converted all the logarithmic probabilities into actual probabilities so that
# they always added to 1 for the function to work properly. In case of a word or tag not being available
# in the emission or transition probabilities respectively, we assigned a very low probability of 1 over 1M.
# As the intial state of tags for each word in a sentence, first we assigned each word the tag of a noun.
# This worked well for us as the samples ultimately converged in about 1000 iterations and we used 1500 samples
# post that to compute the final tags for every sentence. However, this was taking about 20 minutes to
# compute tags for all the 2000 sentences.
# Hence, in order to reduce the time, we modified the initial state of tags to be the one we obtain from
# the Simplified model. Further, we also made sure that words that are not present in the training dataset
# are hard-tagged as nouns, and hence are not sampled at any iteration during MCMC. This helped a LOT in reducing
# the warm-up time required for MCMC as well as improved the accuracy of the model! As a final touch, we modified
# the program to dynamically decide the warm-up period for each sentence - the longer the sentence, the more iterations
# it needs to converge. Therefore, we used 20x as the factor for warm-up period, and 55x as the factor for posterior samples
# leading to a total of 75x iterations for every sentence. Using these two tricks, we were able to reduce the total time taken
# for 2000 sentences to less than 5 minutes from 20 minutes - YUUUGE WIN AGAIN!!
# NOTE: We tried using HMM tags as the initial state, but that did not lead to any major improvement in accuracy and tags.
# Results as per MCMC:
# Words correct: ~94.18%       Sentences correct: ~48.80%
####

import random
import math
import numpy as np


class Solver:

    def posterior(self, model, sentence, label):
        if model == "Simple":
            prob = 0

            # Add each emission probability of word given tag with each other
            # after multiplying each one of them by -1 because we have taken the negative log
            # of the probabilities, while we need to report the positive log
            for i in range(0, len(sentence) ):
                word = sentence[i]
                tag = label[i]
                prob += -P_emission[tag][word] if word in P_emission[tag] else -1000000

            return prob
        elif model == "Complex":
            prob = 0

            # Add each emission probability of word given tag, with
            # the transition probabilities from each tag to the next tag, with
            # the transition probabilities from each tag to the next-to-next tag
            # after multiplying each one of them by -1 because we have taken the negative log
            # of the probabilities, while we need to report the positive log

            # Run a loop starting from the first word until the third last word due to
            # transition constraints on the last two words
            for i in range(0, len(sentence)-2 ):
                word = sentence[i]
                tag = label[i]
                next_tag = label[i+1]
                next_to_next_tag = label[i+2]
                prob += -P_emission[tag][word] if word in P_emission[tag] else -1000000 + \
                        -P_transition[tag][next_tag] if next_tag in P_transition[tag] else -1000000 + \
                        -P_transition_2[tag][next_to_next_tag] if next_to_next_tag in P_transition_2[tag] else -1000000

            # Add emission probability for the last two words that were skipped in the loop above
            # due to constraints on transitions

            # Second last word
            i = len(sentence)-2
            word = sentence[i]
            tag = label[i]
            next_tag = label[i+1]
            prob += -P_emission[tag][word] if word in P_emission[tag] else -1000000 + \
                    -P_transition[tag][next_tag] if next_tag in P_transition[tag] else -1000000

            # Last word
            i = len(sentence)-1
            word = sentence[i]
            tag = label[i]
            prob += -P_emission[tag][word] if word in P_emission[tag] else -1000000

            return prob
        elif model == "HMM":
            # initialize probability with the initial state probability
            tag = label[0]
            prob = -P_initial_state[tag]

            # Add each emission probability of word given tag, with
            # the transition probabilities from each tag to the next tag
            # after multiplying each one of them by -1 because we have taken the negative log
            # of the probabilities, while we need to report the positive log

            # Run a loop starting from the first word until the third last word due to
            # transition constraints on the last two words
            for i in range(0, len(sentence)-1 ):
                word = sentence[i]
                tag = label[i]
                next_tag = label[i+1]
                prob += -P_emission[tag][word] if word in P_emission[tag] else -1000000 + \
                        -P_transition[tag][next_tag] if next_tag in P_transition[tag] else -1000000

            # Add emission probability for the last word that was skipped in the loop above
            # due to no transition from the last word to any other word
            i = len(sentence)-1
            word = sentence[i]
            tag = label[i]
            prob += -P_emission[tag][word] if word in P_emission[tag] else -1000000

            return prob
        else:
            print("Unknown algo!")

    # Do the training!
    def train(self, data):
        # t0 = time()
        # Declare global hashes to store the final probabilities
        global P_initial_state
        global P_transition
        global P_transition_2
        global P_emission
        P_initial_state, P_transition, P_transition_2, P_emission = {}, {}, {}, {}

        # Counter for sentences
        cnt = 0
        # Local hashes to store frequencies for each word separately
        initial_state, transition, transition_2, emission = {}, {}, {}, {}

        for (s, gt) in data:
            # Initialize local hashes for this sentence
            initial_state[cnt], transition[cnt], transition_2[cnt], emission[cnt] = {}, {}, {}, {}

            # Store frequencies for initial states
            initial_state[cnt][ gt[0] ] = (initial_state[cnt][ gt[0] ] + 1) if gt[0] in initial_state[cnt].keys() else 1

            # Loop over the entire sentence to store transition, transition_2, and emission probabilities
            # Note: The loop skips the last two words in the sentence as transition_2 is not defined
            for i in range(0, len(gt)-2):
                if gt[i] in transition[cnt].keys():
                    transition[cnt][ gt[i] ][ gt[i+1] ] = (transition[cnt][ gt[i] ][ gt[i+1] ] + 1) if gt[i+1] in transition[cnt][ gt[i] ].keys() else 1
                    transition_2[cnt][ gt[i] ][ gt[i+2] ] = (transition_2[cnt][ gt[i] ][ gt[i+2] ] + 1) if gt[i+2] in transition_2[cnt][ gt[i] ].keys() else 1
                    emission[cnt][ gt[i] ][ s[i] ] = (emission[cnt][ gt[i] ][ s[i] ] +1) if s[i] in emission[cnt][ gt[i] ].keys() else 1
                else:
                    transition[cnt][ gt[i] ], transition_2[cnt][ gt[i] ], emission[cnt][gt[i] ] = {}, {}, {}
                    transition[cnt][  gt[i] ][ gt[i+1] ], transition_2[cnt][  gt[i] ][ gt[i+2] ], emission[cnt][ gt[i] ][ s[i] ] = 1, 1, 1

            # Add the transition and emission probabilities corr to the last two words separately
            # because the loop above skips the last two words in a sentence.

            # Second-last word
            i = len(gt)-2
            if gt[i] in transition[cnt].keys():
                transition[cnt][ gt[i] ][ gt[i+1] ] = (transition[cnt][ gt[i] ][ gt[i+1] ] + 1) if gt[i+1] in transition[cnt][ gt[i] ].keys() else 1
                emission[cnt][ gt[i] ][ s[i] ] = (emission[cnt][ gt[i] ][ s[i] ]+1) if s[i] in emission[cnt][ gt[i] ].keys() else 1
            else:
                transition[cnt][ gt[i] ], emission[cnt][ gt[i] ] = {}, {}
                transition[cnt][  gt[i] ][ gt[i+1] ], emission[cnt][ gt[i] ][ s[i] ] = 1, 1

            # Last word
            i = len(gt)-1
            if gt[i] in emission[cnt].keys():
                emission[cnt][ gt[i] ][ s[i] ] = (emission[cnt][ gt[i] ][ s[i] ]+1) if s[i] in emission[cnt][ gt[i] ].keys() else 1
            else:
                emission[cnt][ gt[i] ] = {}
                emission[cnt][ gt[i] ][ s[i] ] = 1

            # Update the counter for sentences
            cnt += 1

        # Merge local hashes into global hashes
        for i in range(0, cnt):
            # Merge initial state frequencies
            for key, value in initial_state[i].items():
                P_initial_state[key] = (P_initial_state[key] + value) if key in P_initial_state.keys() else value

            # Merge transition state frequencies
            for key, kv in transition[i].items():
                if key in P_transition.keys():
                    for k, v in kv.items():
                        P_transition[key][k] = (P_transition[key][k] + v) if k in P_transition[key] else v
                else:
                    P_transition[key] = {}
                    for k, v in kv.items():
                        P_transition[key][k] = v

            # Merge transition_2_2 state frequencies
            for key, kv in transition_2[i].items():
                if key in P_transition_2.keys():
                    for k, v in kv.items():
                        P_transition_2[key][k] = (P_transition_2[key][k] + v) if k in P_transition_2[key] else v
                else:
                    P_transition_2[key] = {}
                    for k, v in kv.items():
                        P_transition_2[key][k] = v

            # Merge emission state frequencies
            for key, kv in emission[i].items():
                if key in P_emission.keys():
                    for k, v in kv.items():
                        P_emission[key][k] = (P_emission[key][k] + v) if k in P_emission[key] else v
                else:
                    P_emission[key] = {}
                    for k, v in kv.items():
                        P_emission[key][k] = v

        # Normalize frequencies into probabilities
        P_initial_state = { key: -math.log( val / tot, 10 ) for tot in ( sum(P_initial_state.values() ), ) for key, val in P_initial_state.items() }
        P_transition = { key: { k: -math.log( v / tot, 10 ) for k, v in value.items() } for key, value in P_transition.items() for tot in ( sum(P_transition[key].values() ), ) }
        P_transition_2 = { key: { k: -math.log( v / tot, 10 ) for k, v in value.items() } for key, value in P_transition_2.items() for tot in ( sum(P_transition_2[key].values() ), ) }
        P_emission = { key: { k: -math.log( v / tot, 10 ) for k, v in value.items() } for key, value in P_emission.items() for tot in ( sum(P_emission[key].values() ), ) }

    def simplified(self, sentence):
        global sampled
        pred_labels = []
        cnt = 0
        sampled = []

        for word in sentence:
            # Initialize maximum probability and predicted tag for the word
            max_prob, max_tag = math.inf, ''

            # Loop over all possible tags to identify the one that maximizes the P( word | tag )
            for tag in P_emission.keys():
                # Assign a very low probability in case the word is not found in the training data for a particular tag
                prob = P_emission[tag][word] if word in P_emission[tag].keys() else 1000000
                (max_prob, max_tag) = (prob, tag) if prob < max_prob else (max_prob, max_tag)

            # Assign the predicted tag to be noun in case the word is not found in any of the tags in the training data
            (max_tag, sampled) = ( 'noun', sampled ) if max_prob == 1000000 else (max_tag, sampled+[cnt])
            pred_labels += [ max_tag ]
            cnt += 1

        return pred_labels

    def complex_mcmc(self, sentence):
        global sampled
        pred_labels = []
        mcmc_samples = []
        cnt = 1

        # Initialize a cutoff for warm-up
        cutoff = 20*len(sentence)

        # Initialize the hash to store frequency of tags for each word in the sentence
        freq = {}
        for i in range( 0, len(sentence) ):
            freq[i] = {}

        # define initial state
        # state = [ "noun" for i in range( 0,len(sentence) ) ]
        state = self.simplified(sentence)
        # state = self.hmm_viterbi(sentence)

        # Use MCMC and start sampling
        while cnt < 75*len(sentence):
            # choose and random word to sample from keeping tags for other words fixed
            sample_index = np.random.choice(sampled, 1)[0]
            word = sentence[sample_index]

            # Initialize vars to store all possible tags and corresponding probabilities for the word being sampled
            tags = []
            probs = []

            # Use the provided model to calculate probabilities for each tag
            # Note: Convert all logarithmic probabilities to actual probabilities so that
            # they can be used in the np.random.choice function
            for tag in P_emission.keys():
                # P(tag | word) = P(word | tag) ...
                prob = math.pow(10, -P_emission[tag][word]) if word in P_emission[tag] else 1/1000000

                # if the sampled word is not the first or second word in the sentence
                if sample_index > 1:
                    # ... * P(transition from prev tag to this tag) * P(transition from the tag before to the previous tag)
                    p_word, p_tag, pp_word, pp_tag = sentence[ sample_index-1 ], state[ sample_index-1 ], sentence[ sample_index-2 ], state[ sample_index-2 ]

                    prob *= math.pow(10, -P_transition[p_tag][tag]) if tag in P_transition[p_tag] else 1/1000000 * \
                            math.pow(10, -P_transition_2[pp_tag][tag]) if tag in P_transition_2[pp_tag] else 1/1000000
                # else if the word is the second word
                elif sample_index == 1:
                    p_word, p_tag = sentence[ sample_index-1 ], state[ sample_index-1 ]

                    # ... * P(transition from prev tag to this tag)
                    prob *= math.pow(10, -P_transition[p_tag][tag]) if tag in P_transition[p_tag] else 1/1000000

                # store the tags and corr probabilities
                tags += [tag]
                probs += [prob]

            # Normalize probabilities
            probs[:] = [ prob/sum(probs) for prob in probs ]

            # change current state by flipping a biased coin as per the probabilities calculated
            state = state[:sample_index] + [ np.random.choice( tags, 1, p=probs )[0] ] + state[sample_index+1:]

            # update the frequency of tags for each word as per the current state if the warm-up period is over
            if cnt >= cutoff:
                # add the posterior sample to the list from which we can pick the five samples that need to showcased
                mcmc_samples += [state]
                for i in range(0,len(sentence)):
                    freq[i][ state[i] ] = ( freq[i][ state[i] ] + 1 ) if state[i] in freq[i] else 1

            # move onto the next sample...
            cnt += 1

        # for each word in the sentence, identify the tag with maximum frequency for each tag recorded after the warm-up period
        for pos, tags in freq.items():
            pred_labels += [ max( freq[pos], key=lambda x: freq[pos][x] ) ]

        # print five sampled particles from the posterior distribution
        samples = np.random.choice( range(0, len(mcmc_samples) ), 5 )[:5]
        print("Five sampled particles:")
        for i in samples:
            print( mcmc_samples[i] )

        return pred_labels

    def hmm_viterbi(self, sentence):
        pred_labels, viterbi_table, viterbi_max = [], {}, {}

        for i in range(0, len(sentence)):
            # Initialize maximum probability and corr previous tag for the word
            word, viterbi_table[i], viterbi_max[i] =  sentence[i], {}, {}

            # Loop over all possible tags to identify the one that maximizes the P( word | tag )
            for tag in P_emission.keys():
                max_prev_prob, max_prev_tag = math.inf, ''
                # Assign a very low probability in case the word is not found in the training data for a particular tag
                prob = P_emission[tag][word] if word in P_emission[tag].keys() else 1000000

                if i > 0:
                    # For each tag in the previous state, multiply its probability (stored in viterbi_table) with
                    # the probability of transition to the current tag. Identify the previous tag with maximum
                    # product, store it in viterbi_max, and multiply this maximum product with the emission probability calculated above.

                    for prev_tag in viterbi_table[i-1].keys():
                        prev_prob = ( viterbi_table[i-1][prev_tag] + P_transition[prev_tag][tag] ) if tag in P_transition[prev_tag].keys() else ( viterbi_table[i-1][prev_tag] + 1000000 )
                        (max_prev_prob, max_prev_tag) = (prev_prob, prev_tag) if prev_prob < max_prev_prob else (max_prev_prob, max_prev_tag)

                    viterbi_max[i][tag], viterbi_table[i][tag] = max_prev_tag, (prob + max_prev_prob)

                else:
                    '''No comparison to check if it existis in intial?? : Comment by Nitesh on 21/11/2018 5:57 PM'''
                    viterbi_table[i][tag] = prob + P_initial_state[tag]

            rev_labels = [ min(viterbi_table[i], key=lambda key: viterbi_table[i][key]) ]

            for j in range(i-1,-1,-1):
                next_label = viterbi_max[j+1][ rev_labels[-1] ]
                rev_labels += [next_label]

            pred_labels = rev_labels[::-1]

        return pred_labels

    # solve() method returns a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
