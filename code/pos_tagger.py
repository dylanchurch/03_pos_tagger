import argparse
import collections
import math
import operator
import random

import utils


def create_model(sentences):
    ## Data structures to store the model.
    ## You can modify this data structures if you want to
    prior_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    priors = collections.defaultdict(lambda: collections.defaultdict(float))
    likelihood_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    likelihoods = collections.defaultdict(lambda: collections.defaultdict(float))

    majority_tag_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    majority_baseline = collections.defaultdict(lambda: "NN")

    tag_counts = collections.defaultdict(int)

    # TODO: Create the model (counts) for the majority baseline.
    #   This model only needs to store the most frequent tag for each word


    # TODO: Create te model for the HMM model
    #   You will need to return the prior and likelihood probabilities (and possibly other stuff).
    #   At the end of the day, you need to return what you need in predict_tags(...)
    #   Decide what to smooth, whether you need log probabilities, etc.

    ## You can modify the return value if you want to
    return priors, likelihoods, majority_baseline, tag_counts


def predict_tags(sentences, model, mode='always_NN'):
    priors, likelihoods, majority_baseline, tag_counts = model

    for sentence in sentences:
        if mode == 'always_NN':
            # Do NOT change this one... it is a baseline
            for token in sentence:
                token.tag = "NN"
        elif mode == 'majority':
            # Do NOT change this one... it is a (smarter) baseline
            for token in sentence:
                token.tag = majority_baseline[token.word]
        elif mode == 'hmm':
            # TODO The bulk of your code goes here
            #   1. Create the Viterbi Matrix
            #   2. Fill the Viterbi matrix
            #      You will need one loop to fill the first column
            #      and a triple nested loop to fill the remaining columns
            #   3. Recover the sequence of tags and update token.tag accordingly
            # The current implementation tags everything as an NN you need to change it
            for token in sentence:
                token.tag = "NN"
        else:
            assert False

    return sentences


if __name__ == "__main__":
    # Do NOT change this code (the main method)
    parser = argparse.ArgumentParser()
    parser.add_argument("PATH_TR",
                        help="Path to train file with POS annotations")
    parser.add_argument("PATH_TE",
                        help="Path to test file (POS tags only used for evaluation)")
    parser.add_argument("--mode", choices=['always_NN', 'majority', 'hmm'], default='always_NN')
    args = parser.parse_args()

    tr_sents = utils.read_tokens(args.PATH_TR) #, max_sents=1)
    # test=True ensures that you do not have access to the gold tags (and inadvertently use them)
    te_sents = utils.read_tokens(args.PATH_TE, test=True)

    model = create_model(tr_sents)

    print("** Testing the model with the training instances (boring, this is just a sanity check)")
    gold_sents = utils.read_tokens(args.PATH_TR)
    predictions = predict_tags(utils.read_tokens(args.PATH_TR, test=True), model, mode=args.mode)
    accuracy = utils.calc_accuracy(gold_sents, predictions)
    print(f"[{args.mode:11}] Accuracy "
          f"[{len(list(gold_sents))} sentences]: {accuracy:6.2f} [not that useful, mostly a sanity check]")
    print()

    print("** Testing the model with the test instances (interesting, these are the numbres that matter)")
    # read sentences again because predict_tags(...) rewrites the tags
    gold_sents = utils.read_tokens(args.PATH_TE)
    predictions = predict_tags(te_sents, model, mode=args.mode)
    accuracy = utils.calc_accuracy(gold_sents, predictions)
    print(f"[{args.mode}:11] Accuracy "
          f"[{len(list(gold_sents))} sentences]: {accuracy:6.2f}")
