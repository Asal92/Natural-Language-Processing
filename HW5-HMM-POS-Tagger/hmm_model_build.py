from nltk.corpus import brown
import pickle
import numpy as np
import re
import sys


START_STR = "<s>"
END_STR = "</s>"
UNKNOWN_STR = "unk"


def uni_tag_count(sents):
    uni_count_dict = {}
    uni_count_dict[START_STR] = len(sents)
    for s in sents:
        for t in s: # t is a tuple
            if t[1] in uni_count_dict:
                uni_count_dict[t[1]] += 1
            elif t[1] not in uni_count_dict:
                uni_count_dict[t[1]] = 1
    # Adding start and end tags in my count
    uni_count_dict[END_STR] = len(sents)
    return uni_count_dict


def bi_tag_count(sents):
    bi_count_dict = {}
    for s in sents:
        l = len(s)
        # if index == 0: add start tag, and if index == len(s)-1: add end tag
        for i in range(0,l):
            bi_tag = []
            tag = s[i][1]

            if i == 0:
                # add start + add with next element
                if l > 1: # what if len is 1?
                    bi_tag.append((START_STR,tag))
                    tag1 = s[i + 1][1]
                    bi_tag.append((tag,tag1))
                else:
                    bi_tag.append((START_STR, tag))
                    bi_tag.append((tag, END_STR))
            elif i == l-1:
                # add end
                bi_tag.append((tag,END_STR))
            else:
                # do normal i:i+1
                tag1 = s[i + 1][1]
                bi_tag.append((tag,tag1))

            for b in bi_tag:
                if b in bi_count_dict:
                    bi_count_dict[b] += 1
                else:
                    bi_count_dict[b] = 1
    return bi_count_dict

def build_vocab(sents):
    # read the book for unknowns. I think this for testing not training!!
    v = {}
    for s in sents:
        for t in s:
            if t[0] not in v.keys():
                v[t[0]] = 1
            else:
                v[t[0]] += 1
    return v


def word_tag_count(sents):
    word_tag_count_dict = {}
    for s in sents:
        for t in s:
            if t in word_tag_count_dict.keys():
                word_tag_count_dict[t] += 1
            else:
                word_tag_count_dict[t] = 1
    return word_tag_count_dict


def transition_matrix(uni_count, bi_count):
    # 474 x 474 matrix (rows = ti-1, cols = ti)
    A={} # making a dict instead
    for i in uni_count.keys():
        for j in uni_count.keys():
            try:
                A[(i,j)] = np.log(float(bi_count[(i,j)] / uni_count[i]))
            except KeyError: # bs log(0) is undefined
                A[(i, j)] = -np.inf
    return A


def emission_matrix(word_tag_count, uni_count, vocab):
    # 474 x num_of_words
    B = {}
    for i in uni_count.keys():
        for j in vocab.keys():
            try:
                B[(i,j)] = np.log(float(word_tag_count[(j,i)]/uni_count[i]))
            except KeyError:
                B[(i, j)] = -np.inf
    return B


def model_build(corpus_module):
    '''
    1. load brown corpus
    2. count unigram, bigran tags + word to tags
    3. count probabilities A[p(t|t)] and B[p(w|t)]
    4. write them back to a .dat file
    '''

    fm = open(corpus_module, 'rb')
    sents = pickle.load(fm) #list of lists of tuples
    #sents = brown.tagged_sents()
    print("length of brown corpus sentences: ", len(sents))
    uni_dict = uni_tag_count(sents)#dict of tags
    print("length of tags: ", len(uni_dict))
    bi_dict = bi_tag_count(sents)#dict of tuples
    print("length of bigrams: ", len(bi_dict))
    word_tag_dict = word_tag_count(sents)#dict of tuples(word-tag)
    print("length of word-tag counts: ", len(word_tag_dict))
    V = build_vocab(sents)#dict
    print("length of brown corpus vocabulary: ", len(V))
    A = transition_matrix(uni_dict, bi_dict)#dict of tuples (ti-1,ti)
    print("length of transition dictionary: ", len(A))
    B = emission_matrix(word_tag_dict,uni_dict,V)# dict of tuples (t,w)
    print("length of emission dictionary: ", len(B))
    print("now dumping transition, emission and vocabulary in model...")
    f = open("model.dat",'wb')
    pickle.dump(A, f) # bigram probability
    pickle.dump(B, f) # word tag probability
    pickle.dump(V, f) # vocabulary
    pickle.dump(uni_dict, f) # unigrams
    f.close()
    print("model is created.")




if __name__ == '__main__':
    
    # I am using brown_untagged_sents.dat
    #f_corpus = sys.argv[1]
    f_corpus = "brown_tagged_sents.dat"
    model_build(f_corpus)




