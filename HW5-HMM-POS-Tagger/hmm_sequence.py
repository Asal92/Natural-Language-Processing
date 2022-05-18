from nltk.corpus import brown
import pickle
import numpy as np
import re
import sys

UNK_ONE_OCCURRENCE = True
UNK_OVER_TAGS = False

START_STR = "<s>"
END_STR = "</s>"
UNKNOWN_STR = "unk"
UNK_PROB = np.log(float(1/474))


def sequence(file_model, file_text):
    ft = open(file_text, 'r')
    tagged_sent = ft.read()
    tagged_sent = tagged_sent.replace("\n", "")
    if tagged_sent[-1] == " ":# what if there is space at the end of the sentence??!!
        tagged_sent = tagged_sent[: -1]
    word_sequence = tagged_sent.split(' ')
    print("here is my input:")
    print(word_sequence)
    word_seq_list = []
    tag_seq_list = []
    for s in word_sequence:
        word_seq_list.append(re.sub(r'(.*)/(.*)', r'\1', s))
        tag_seq_list.append(re.sub(r'(.*)/(.*)', r'\2', s))
    print("list of words:", word_seq_list)
    print("list of tags:", tag_seq_list)

    print("loding the model..")
    fm = open(file_model,'rb')
    # if I dumped 3 files, I have to call load 3 times also
    A = pickle.load(fm) # transition
    B = pickle.load(fm) # emission
    V = pickle.load(fm) # vocabulary
    one_v = [] # words that have appeared once in the corpus
    for key,value in V.items():
        if value == 1:
            one_v.append(key)

    # p(tag sequence) = p(w|t)p(ti|ti-1)
    prob=0
    B_prob=0
    for i in range(len(word_sequence)):
        # handling unknown word by assigning constant probability (1/474) to it (Book Unknown words suggestion)
        if UNK_OVER_TAGS is True:
            if word_seq_list[i] not in V.keys():
                B_prob = UNK_PROB
            else:
                B_prob = B[(tag_seq_list[i], word_seq_list[i])]
        # handling unknown word by replacing it with words that have occurred once in the corpus with that tag
        if UNK_ONE_OCCURRENCE is True:
            if word_seq_list[i] not in V.keys():
                for w in one_v:
                    if B[(tag_seq_list[i], w)] != -np.inf:
                        B_prob = B[(tag_seq_list[i], w)]
                        break
            else:
                B_prob = B[(tag_seq_list[i], word_seq_list[i])]

        if i==0:
            prob += B_prob + A[(START_STR, tag_seq_list[i])]
        elif i==len(word_sequence)-1:
            prob += B_prob + A[(tag_seq_list[i-1],tag_seq_list[i])]
            prob += A[(tag_seq_list[i],END_STR)]
        else:
            prob += B_prob + A[(tag_seq_list[i - 1], tag_seq_list[i])]
    print("HMM sequence probability calculation is:", prob)


if __name__ == '__main__':
    # 2 command line arguments: model file name + text file name (single tagged sentence)
    # load the model file
    # use the model file to generate prob for the sequence using bigram tags and word to tag count
    f_model = sys.argv[1]
    f_text = sys.argv[2]
    #f_model = "model.dat"
    #f_text = "testfile.txt"
    sequence(f_model, f_text)
    # expect larg negative number
