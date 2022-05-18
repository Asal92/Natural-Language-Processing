from nltk.corpus import brown
import pickle
import numpy as np
import re
import sys

UNK_ONE_OCCURRENCE = True # if you wanna use words that have occured once instead of unknown words
UNK_OVER_TAGS = False # if you just wanna use 1/474 equal probability for all tags for unknown words

START_STR = "<s>"
END_STR = "</s>"
UNKNOWN_STR = "unk"
UNK_PROB = np.log(float(1/474))


def viterbi(A, B, V, Q, W):
    '''
    Viterbi algorithm
    :param A: dict. Transition Matrix
    :param B: dict. Emission Matrix
    :param V: list. Brown corpus vocabulary
    :param Q: list. of brown corpus tags
    :param W: list. of untagged words to find tag for
    :return: tag predictions
    '''
    # list of dictionaries, each initial dictionary is a state(TAG) and for each initial state,
    # I have another dictionary including probability and previous state
    VIT = [{}]
    B_prob=0
    one_v = []  # words that have appeared once in the corpus
    for key, value in V.items():
        if value == 1:
            one_v.append(key)

    # Initiation
    for st in Q:
        # handling unknown word by assigning constant probability (1/474) to it (Book Unknown words suggestion)
        if UNK_OVER_TAGS is True:
            if W[0] not in V.keys():
                B_prob = UNK_PROB
            else:
                B_prob = B[(st, W[0])]
        # handling unknown word by replacing it with words that have occurred once in the corpus with that tag
        if UNK_ONE_OCCURRENCE is True:
            if W[0] not in V.keys():
                for w in one_v:
                    if B[(st, w)] != -np.inf:
                        B_prob = B[(st, w)]
                        break
            else:
                B_prob = B[(st, W[0])]
        VIT[0][st] = {"prob": A[(START_STR, st)] + B_prob, "prev": None}

    # Recursion + Termination (I've included end tag also here)
    for t in range(1, len(W)):
        VIT.append({})
        for st in Q:
            max_tr_prob = VIT[t - 1][Q[0]]["prob"] + A[(Q[0], st)]
            prev_st_selected = Q[0]
            for prev_st in Q[1:]:
                tr_prob = VIT[t - 1][prev_st]["prob"] + A[(prev_st, st)]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            # handling unknown word by assigning constant probability (1/474) to it (Book Unknown words suggestion)
            if UNK_OVER_TAGS is True:
                if W[t] not in V.keys():
                    B_prob = UNK_PROB
                else:
                    B_prob = B[(st, W[t])]
            # handling unknown word by replacing it with words that have occurred once in the corpus with that tag
            if UNK_ONE_OCCURRENCE is True:
                if W[t] not in V.keys():
                    for w in one_v:
                        if B[(st, w)] != -np.inf:
                            B_prob = B[(st, w)]
                            break
                else:
                    B_prob = B[(st, W[t])]
            max_prob = max_tr_prob + B_prob
            VIT[t][st] = {"prob": max_prob, "prev": prev_st_selected}

    TAG_SEQ = []
    max_prob = -np.inf
    best_st = None

    #looking at last state only from viterbi
    for st, data in VIT[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    TAG_SEQ.append(best_st)
    BACK = best_st
    # Now using last state and backtracking to the most likely TAG
    for t in range(len(VIT) - 2, -1, -1):
        TAG_SEQ.insert(0, VIT[t + 1][BACK]["prev"])
        BACK = VIT[t + 1][BACK]["prev"]

    return TAG_SEQ


if __name__ == '__main__':
    # 2 command line arguments: model file name + text file name (single tagged sentence)
    # text file contains a single untagged sentence
    f_model = sys.argv[1]
    f_text = sys.argv[2]
    #f_model = "model.dat"
    #f_text = "example_untagged.txt"
    ft = open(f_text, 'r')
    untagged_sent = ft.read()
    untagged_sent = untagged_sent.replace("\n", "")
    W = untagged_sent.split(' ')
    print("here is a list of my words:", W)

    print("loading the model...")
    fm = open(f_model, 'rb')
    A = pickle.load(fm) # transition
    B = pickle.load(fm) # emission
    V = pickle.load(fm) # Vocabulary
    U = pickle.load(fm) # unigrams dictionary with values
    Q = list(U.keys())

    print("starting the viterbi calculation...")
    preds = viterbi(A, B, V, Q, W)
    #print("Here is the Viterbi Prediction:")
    #print(preds)
    vit_pred = ""
    for i in range(len(W)):
        vit_pred += W[i]+"/"+preds[i]+" "
    print("***************************************")
    print("And now... HMM Viterbi MLS Calculation: ")
    print(vit_pred)
