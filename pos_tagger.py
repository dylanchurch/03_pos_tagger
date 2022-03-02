import argparse
import collections
import math
import operator
import random

import utils


def create_model(sentences):
    ## Data structures to store the model.
    ## You can modify this data structures if you want to (use these data structs)
    prior_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    priors = collections.defaultdict(lambda: collections.defaultdict(float))#likelihood of next tag being y if previous tag was x
    likelihood_counts = collections.defaultdict(lambda: collections.defaultdict(int))#likelihood of word having a certain tag
    likelihoods = collections.defaultdict(lambda: collections.defaultdict(float))

    majority_tag_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    majority_baseline = collections.defaultdict(lambda: "NN")

    tag_counts = collections.defaultdict(int)#Total number of occurances of each tag

    # TODO: Create the model (counts) for the majority baseline.
    #   This model only needs to store the most frequent tag for each word
    
    #TEST--------------------------------------------
    #for sentence in sentences:
    #    for token in sentence:
    #        print(token.word)
    #Each token consists of the of the form word/tag
    #------------------------------------------------
    prior="$"#Initialize prior
    for sentence in sentences:
        for token in sentence:
            prior_counts[prior][token.tag]+=1#Increment prior counts?
            prior=token.tag#reset prior
            likelihood_counts[token.word][token.tag]+=1#Increment likelihood counts
            majority_tag_counts[token.word][token.tag]+=1#Increment majority tag counts
            tag_counts[token.tag]+=1#increment tag counts

    # TODO: Create te model for the HMM model
    #   You will need to return the prior and likelihood probabilities (and possibly other stuff).
    #   At the end of the day, you need to return what you need in predict_tags(...)
    #   Decide what to smooth, whether you need log probabilities, etc.
    for tag1 in priors:#iterate through prior dictionary
        for tag2 in tag1:
            priors[tag1][tag2]=prior_counts[tag1][tag2]/tag_counts[tag2]#Calculate prior likelihood
            likelihoods[tag1][tag2]=likelihood_counts[tag1][tag2]/tag_counts[tag2]#Calculate likelihoods

    for word in majority_tag_counts:
        max=0
        tag_max="NN"
        for tag in majority_tag_counts[word]:
            if majority_tag_counts[word][tag]>max:
                max=majority_tag_counts[word][tag]#Reasign max
                tag_max=tag#Reassign tag_max
        majority_baseline[word]=tag_max#Assign tag to majority baseline

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
