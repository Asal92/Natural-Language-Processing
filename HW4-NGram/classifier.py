import sys
import numpy as np
import nltk
from nltk.corpus import words
import re
import operator
import random

START_STR = "<s>"
END_STR = "</s>"
UNKNOWN_STR = "unk"
SMOOTHING_THRESH = 6
TRIGRAM = True
lambda1 = float(0.1)
lambda2 = float(0.2)
lambda3 = float(0.7)


class Ngram_Model:
    def __init__(self, file):
        self.file = file
        self.sentences= self.sent_tokenize()
        self.sent_len = len(self.sent_tokenize())  # for computing number of starts/ends
        self.tokens = self.word_tokenize()
        self.vocabulary, self.word_frequency = self.build_vocab(self.tokens)
        self.vocabulary_length = len(self.vocabulary)

    def sent_tokenize(self):
        f = open(self.file.name, encoding="UTF-8")
        result = []

        result = nltk.tokenize.sent_tokenize(f.read().lower())
        result = [[i] for i in result]
        for r in result:
            s = r[0]
            s = re.sub(r'[’“`´]', r'', s)
            s = re.sub(r'[`™£¢∞§¶•ªº–≠«‘“πøˆ¨¥†®´œåß∂ƒ©˙∆˚¬…æ≥≤µ˜√ç≈Ω]', r'', s)
            s.replace("’", "")
            s.replace('“', "")
            s = re.sub(r'[\'\"]', r'', s)
            s = re.sub(r'--', r' ', s)
            s = re.sub(r'-', r' ', s)
            s = re.sub(r'[\.\?!,\'\":;_#$@&*%^~<>(){}\[\]—]', r'', s)  # /\
            s = re.sub(r'([0-9]*)', r'', s)
            result[result.index(r)] = [s]

        return result

    def word_tokenize(self):
        f = open(self.file.name, encoding="UTF-8")
        result = []
        text = str(f.read())
        text = text.lower()
        text = re.sub(r'[’“`´]', r'', text)
        text = re.sub(r'[`™£¢∞§¶•ªº–≠«‘“πøˆ¨¥†®´œåß∂ƒ©˙∆˚¬…æ≥≤µ˜√ç≈Ω]', r'', text)
        text.replace("’", "")
        text.replace('“', "")
        text = re.sub(r'[\'\"]', r'', text)
        text = re.sub(r'--', r' ', text)
        text = re.sub(r'-', r' ', text)
        text = re.sub(r'[\.\?!,:;_#$@&*%^~<>(){}\[\]—]', r'', text)  # /\
        text = re.sub(r'([0-9]*)', r'', text)

        result = nltk.tokenize.word_tokenize(text, language='english')
        return result

    def build_vocab(self, tokens):
        vocabulary = {}  # make the vocabulary a dictionary!!!
        word_freq = {}
        dictionary = set(words.words())
        unk_tokens = {}

        vocabulary[START_STR] = 1  # start
        vocabulary[END_STR] = 1  # end
        vocabulary[UNKNOWN_STR] = 1  # not sure if I should add unknown to vocabulary

        for t in tokens:
            if t in dictionary:
                if t not in vocabulary:
                    vocabulary[t] = 1
                if t in word_freq:
                    word_freq[t] += 1
                else:
                    word_freq[t] = 1
            else:
                if t in unk_tokens:
                    unk_tokens[t] += 1
                else:
                    unk_tokens[t] = 1

        word_freq[UNKNOWN_STR] = len(unk_tokens)

        sorted_word_freq = dict(
            sorted(word_freq.items(), key=operator.itemgetter(1), reverse=False))  # descending or ascending??
        return vocabulary, sorted_word_freq

    def bigram_freq(self, bigram_list):
        bigram_freq = {}
        for bigram in bigram_list:
            b1 = str(bigram[0])
            b2 = str(bigram[1])
            bigram_str = b1 + ' ' + b2
            if bigram_str in bigram_freq:
                bigram_freq[bigram_str] += 1
            else:
                bigram_freq[bigram_str] = 1

        sorted_bigram_freq = dict(sorted(bigram_freq.items(), key=operator.itemgetter(1), reverse=True))
        return sorted_bigram_freq

    def trigram_freq(self, trigram_list):
        trigram_freq = {}
        for trigram in trigram_list:
            t1 = str(trigram[0])
            t2 = str(trigram[1])
            t3 = str(trigram[2])
            trigram_str = t1 + ' ' + t2 + ' ' + t3
            if trigram_str in trigram_freq:
                trigram_freq[trigram_str] += 1
            else:
                trigram_freq[trigram_str] = 1

        sorted_trigram_freq = dict(sorted(trigram_freq.items(), key=operator.itemgetter(1), reverse=True))
        return sorted_trigram_freq

    def bigram_prob(self, bigram, bigram_freq, word_freq):
        bigram_text = ' '.join(bigram)
        try:
            bg_freq = bigram_freq[bigram_text]
            bg_freq = int(0 if bg_freq is None else bg_freq)
        except KeyError:
            bg_freq = 0

        if bigram[0] == '<s>':
            sfreq = self.sent_len # if bigram contains start of string, count number of sentences to be it's count
        else:
            sfreq = word_freq[bigram[0]]

        try:
            prob = np.log(bg_freq / sfreq)
        except ZeroDivisionError: # we shouldn't get this error, but just in case!!
            prob=0
        return prob

    def trigram_prob(self, trigram, trigram_freq, bigram_freq):
        trigram_text = ' '.join(trigram)
        try:
            trg_freq = trigram_freq[trigram_text]
            trg_freq = int(0 if trg_freq is None else trg_freq)
        except KeyError:
            trg_freq = 0

        if trigram[0] == '<s>': # this to counts the quotient
            bfreq = self.sent_len  # if trigram contains start of string, count number of sentences to be it's count
        else:
            bfreq = bigram_freq[' '.join(trigram[0:len(trigram)-1])]
            bfreq = int(0 if bfreq is None else bfreq)

        try:
            prob = np.log(trg_freq / bfreq)
        except ZeroDivisionError:  # we shouldn't get this error, but just in case!!
            prob = 0
        return prob

    def N_counts(self, ngram_freq):
        N_counts = {}
        max_freq = max(ngram_freq.values())

        for n in range(1,max_freq+1):
            N_counts["N"+str(n)] = sum(x == n for x in ngram_freq.values())

        count_minus_n0 = sum(i for i in N_counts.values())
        N0 = (self.vocabulary_length * self.vocabulary_length) - count_minus_n0
        N_counts["N0"] = N0
        total_count = N0 + count_minus_n0
        return N_counts

    def good_turing(self, ngram, ngram_freq, n_1gram_freq, N_counts):  # This should be for when accounting unseen bigram
        # count_star = bigram count + 1 * Nbigram+1 / N bigram
        if len(ngram) == 2: # bigram
            if ngram[0] not in self.vocabulary:
                ngram = UNKNOWN_STR + ' ' + ngram[1]
            if ngram[1] not in self.vocabulary:
                ngram = ngram[0] + ' ' + UNKNOWN_STR
        elif len(ngram) == 3: # trigram ["a","b","c"]
            if ngram[0] not in self.vocabulary:
                ngram = UNKNOWN_STR + ' ' + ngram[1] + ' ' + ngram[2]
            if ngram[1] not in self.vocabulary:
                ngram = ngram[0] + ' ' + UNKNOWN_STR + ' ' + ngram[2]
            if ngram[2] not in self.vocabulary:
                ngram = ngram[0] + ' ' + ngram[1] + ' ' + UNKNOWN_STR

        ngram_str = ' '.join(ngram)
        try:
            ngram_count = ngram_freq[ngram_str]  # to see which N to use
        except KeyError: # if bigram not in frequency dictionary...
            ngram_count = 0

        try:
            n_1gram_count = n_1gram_freq[' '.join(ngram[0:len(ngram)-1])]
        except KeyError:
            n_1gram_count = 0

        ngram_count_star = ((ngram_count + 1) * N_counts["N"+str(ngram_count + 1)]) / N_counts["N"+str(ngram_count)]
        if n_1gram_count <= SMOOTHING_THRESH: # just in case, otherwise it won't be needed!
            n_1gram_count_star = ((n_1gram_count + 1) * N_counts["N"+str(n_1gram_count + 1)]) / N_counts["N"+str(n_1gram_count)]
        else:
            n_1gram_count_star = n_1gram_count

        try:
            prob = np.log(ngram_count_star / n_1gram_count_star)
        except ZeroDivisionError:
            prob = 0

        return prob



def ngrams(vocabulary, sentences, n):  # input: list of sentences, n for ngram
    # maybe I should do sentence tokenization here instead!!
    n_grams = []

    if len(sentences) > 1: # for when I am passing class's sentences vs dev set sentence
        for s in sentences:
            if len(s) >= 1:
                text = s[0] # it's the only element of a list
                text_split = text.split()  # simple word tokenization based on whitespace
                for i in range(len(text_split)):
                    if text_split[i] not in vocabulary:
                        text_split[i] = UNKNOWN_STR
                    elif text_split[i] in vocabulary and i==UNKNOWN_STR:
                        text_split[i] = UNKNOWN_STR
                for i in range(len(text_split)):
                    if len(text_split) > n-1: # if # of tokens are n-1
                            if i == 0:
                                if n == 2:
                                    n_grams.append(list((START_STR, text_split[i])))
                                    n_grams.append(text_split[i:i + n])
                                if n == 3:
                                    n_grams.append(list((START_STR, text_split[i], text_split[i + 1])))
                                    n_grams.append(text_split[i:i + n])
                            elif i == (len(text_split) - 1) and n == 2:
                                n_grams.append(list((text_split[i], END_STR)))
                            elif i == (len(text_split) - 2) and n == 3:
                                n_grams.append(list((text_split[i], text_split[i + 1], END_STR)))
                            elif i == len(text_split)-1 and n==3:
                                continue
                            else:
                                n_grams.append(text_split[i:i + n])
                    else:
                        if n == 2:
                            n_grams.append(list((text_split[i], END_STR)))
    elif len(sentences) == 1:
        text = sentences[0]
        text_split = text.split()  # split based on whitespace in a new list
        for i in range(len(text_split)):
            if text_split[i] not in vocabulary:
                text_split[i] = UNKNOWN_STR
            elif text_split[i] in vocabulary and i == UNKNOWN_STR:
                text_split[i] = UNKNOWN_STR
        for i in range(len(text_split)):
            if len(text_split) > 1:
                if i == 0:
                    if n == 2:
                        n_grams.append(list((START_STR, text_split[i])))
                        n_grams.append(text_split[i:i + n])
                    if n == 3:
                        n_grams.append(list((START_STR, text_split[i], text_split[i + 1])))
                        n_grams.append(text_split[i:i + n])
                elif i == (len(text_split) - 1) and n == 2:
                    n_grams.append(list((text_split[i], END_STR)))
                elif i == (len(text_split) - 2) and n == 3:
                    n_grams.append(list((text_split[i], text_split[i + 1], END_STR)))
                elif i == len(text_split) - 1 and n == 3:
                    continue
                else:
                    n_grams.append(text_split[i:i + n])
            else:
                if n == 2:
                    n_grams.append(list((text_split[i], END_STR)))
                if n == 3:
                    n_grams.append(list((START_STR, text_split[i], END_STR)))
    return n_grams


def filter_simple_sent(sent):  # filter out punctuations/numbers  <<<expand this
    line = sent
    if line != "\n":  # "\n"
        line = line.lower()
        line = re.sub(r'[’“`´]', r'', line)
        line = re.sub(r'[`™£¢∞§¶•ªº–≠«‘“πøˆ¨¥†®´œåß∂ƒ©˙∆˚¬…æ≥≤µ˜√ç≈Ω]', r'', line)
        line.replace("’", "")
        line.replace('“', "")
        line = re.sub(r'[\'\"]', r'', line)
        line = re.sub(r'--', r' ', line)
        line = re.sub(r'-', r' ', line)
        line = re.sub(r'[\.\?!,\'\":;_#$@&*%^~<>(){}\[\]—]', r'', line)  # /\
        line = re.sub(r'([0-9]*)', r'', line)
    return line


def create_train_dev(file_name):
    # count the number of paragraphs
    f = open("sources/" + file_name, 'r', encoding="UTF-8")
    paragraphcount = 0

    for line in f.readlines():
        if line in ('\n', '\r\n'):
            paragraphcount += 1

    # compute how much is 20%
    dev_perccount = int(paragraphcount * 0.2)
    dev_rand_parg = random.sample(range(0, paragraphcount), dev_perccount)
    # randomly select that 20% and save it to a new text
    # save the rest into train text
    cnt = 0
    dev_text = ""
    train_text = ""
    f = open("sources/" + file_name, 'r', encoding="UTF-8")  # just in case

    for line in f:
        if line in ('\n', '\r\n'):
            cnt += 1
            '''if (cnt-1) in dev_rand_parg:
                dev_text += '\n'
            else:
                train_text += '\n'''
        else:
            if cnt in dev_rand_parg:
                dev_text += line
            else:
                train_text += line

    dev_name = f.name.replace(".txt", "_dev.txt")
    train_name = f.name.replace(".txt", "_train.txt")
    dev_file = open(dev_name, 'w', encoding="UTF-8")
    dev_file.write(dev_text)
    train_file = open(train_name, 'w', encoding="UTF-8")
    train_file.write(train_text)

    return train_name, dev_name # returning the name of the files




def classifier(author_file):

    if TRIGRAM == False:
        print("training... (This may take a while)")
        authorList_train = []
        authorList_dev = []
        classList = []
        bigramList = []
        bigramFreqList = []
        nCountList = []

        # making train/dev sets based on original text file
        with open(author_file, 'r', encoding="UTF-8") as fp:
            for count, line in enumerate(fp):
                s = line.rstrip('\n')
                s_train, s_dev = create_train_dev(s)
                s_train = re.sub(r'sources/(.*)', r'\1', s_train)
                s_dev   = re.sub(r'sources/(.*)', r'\1', s_dev)
                authorList_train.append(s_train)    # train file names
                authorList_dev.append(s_dev)        # dev file name
        # building my language model
        for a in range(count + 1):
            print(f'loading {authorList_train[a]}...')
            f = open("sources/"+authorList_train[a], 'r', encoding="UTF-8")
            classList.append(Ngram_Model(f))
            print(f'loading {authorList_train[a]} bigram...')
            bigramList.append(ngrams(classList[a].vocabulary, classList[a].sentences, 2))
            print(f'loading {authorList_train[a]} bigram frequency...')
            bigramFreqList.append(classList[a].bigram_freq(bigramList[a]))
            print(f'loading {authorList_train[a]} Ncounts... this takes longer than others')
            nCountList.append(classList[a].N_counts(bigramFreqList[a]))


        # now testing dev sets...
        print("loading dev sets now. apologies, this also may take a while! ran out of time to optimize:( ")
        for i in range(count + 1):
            cor_predict_name = authorList_dev[i]
            cor_predict_name = re.sub(r'(.*)_utf8_(.*).txt', r'\1', cor_predict_name)
            dev_set_predictions = {"C":0, "N":0}

            dev_f = open("sources/"+authorList_dev[i], 'r', encoding="UTF-8")
            sent_tokens = nltk.sent_tokenize(dev_f.read().lower())
            #sent_tokens = [[i] for i in sent_tokens]
            for r in sent_tokens:
                s = filter_simple_sent(r)
                sent_tokens[sent_tokens.index(r)] = [s]
            print(len(sent_tokens), dev_set_predictions)
            for s in sent_tokens: # for each sentence
                #print("new sentence...")
                prob_per_sent = []
                for i in range(count + 1): # for each train set
                    bigrams = ngrams(classList[i].vocabulary, [s[0]], 2)
                    prob = 0
                    for bg in bigrams:  # check if it's in class's bigramList
                        if bg in bigramList[i]:
                            bg_count = bigramFreqList[i].get(' '.join(bg))
                            bg_count = int(0 if bg_count is None else bg_count)
                            if bg_count > SMOOTHING_THRESH:
                                # regular count
                                prob += classList[i].bigram_prob(bg, bigramFreqList[i], classList[i].word_frequency)
                            if bg_count <= SMOOTHING_THRESH:
                                # good-turing smoothing
                                prob += classList[i].good_turing(bg, bigramFreqList[i], classList[i].word_frequency,
                                                                 nCountList[i])
                        else:
                            prob += classList[i].good_turing(bg, bigramFreqList[i], classList[i].word_frequency,
                                                             nCountList[i])
                    log_prob = prob
                    prob_per_sent.append(log_prob)
                #print("predict calculation...")
                predict = prob_per_sent.index(max(prob_per_sent))
                predict_name = authorList_train[predict]
                predict_name = re.sub(r'(.*)_utf8_(.*).txt', r'\1', predict_name)

                if cor_predict_name == predict_name:
                    dev_set_predictions["C"] += 1
                else:
                    dev_set_predictions["N"] += 1
            print(dev_set_predictions)
            try:
                cor_percent = float((dev_set_predictions["C"] / (dev_set_predictions["C"] + dev_set_predictions["N"])) * 100)
            except ZeroDivisionError:
                cor_percent = 100  # if there was no wrong prediction!!
            print( f'for author {cor_predict_name}, predicted correctly {dev_set_predictions["C"]} / '
                   f'{dev_set_predictions["C"] + dev_set_predictions["N"]} = {cor_percent}%')

    elif TRIGRAM == True:
        print("training... (This may take a while)")
        authorList_train = []
        authorList_dev = []
        classList = []
        bigramList = []
        bigramFreqList = []
        biNCountList = []
        trigramList = []
        trigramFreqList = []
        triNCountList = []

        # making train/dev sets based on original text file
        with open(author_file, 'r', encoding="UTF-8") as fp:
            for count, line in enumerate(fp):
                s = line.rstrip('\n')
                s_train, s_dev = create_train_dev(s)
                s_train = re.sub(r'sources/(.*)', r'\1', s_train)
                s_dev = re.sub(r'sources/(.*)', r'\1', s_dev)
                authorList_train.append(s_train)  # train file names
                authorList_dev.append(s_dev)  # dev file name
        # building my langauge models
        for a in range(count + 1):
            print(f'loading {authorList_train[a]}...')
            f = open("sources/"+authorList_train[a], 'r', encoding="UTF-8")
            classList.append(Ngram_Model(f))
            print(f'loading {authorList_train[a]} bigram...')
            bigramList.append(ngrams(classList[a].vocabulary, classList[a].sentences, 2))
            print(f'loading {authorList_train[a]} bigram frequency...')
            bigramFreqList.append(classList[a].bigram_freq(bigramList[a]))
            print(f'loading {authorList_train[a]} biNcounts... this takes longer than others')
            biNCountList.append(classList[a].N_counts(bigramFreqList[a]))
            print(f'loading {authorList_train[a]} trigram...')
            trigramList.append(ngrams(classList[a].vocabulary, classList[a].sentences, 3))
            print(f'loading {authorList_train[a]} trigram frequency...')
            trigramFreqList.append(classList[a].trigram_freq(trigramList[a]))
            print(f'loading {authorList_train[a]} trNcounts... this takes longer than others')
            triNCountList.append(classList[a].N_counts(trigramFreqList[a]))

        # now testing dev sets...
        print("loading dev sets now. apologies, this also may take a while! ran out of time to optimize:( ")
        for i in range(count + 1):
            cor_predict_name = authorList_dev[i]
            cor_predict_name = re.sub(r'(.*)_utf8_(.*).txt', r'\1', cor_predict_name)
            dev_set_predictions = {"C": 0, "N": 0}

            dev_f = open("sources/" + authorList_dev[i], 'r', encoding="UTF-8")
            sent_tokens = nltk.sent_tokenize(dev_f.read().lower())
            # sent_tokens = [[i] for i in sent_tokens]
            for r in sent_tokens:
                s = filter_simple_sent(r)
                sent_tokens[sent_tokens.index(r)] = [s]
            print(len(sent_tokens), dev_set_predictions)
            for s in sent_tokens:  # for each sentence
                # print("new sentence...")
                prob_per_sent = []
                for i in range(count + 1):  # for each train set
                    trigrams = ngrams(classList[i].vocabulary, [s[0]], 3)
                    prob = 0
                    for trg in trigrams:  # check if it's in class's bigramList
                        bg = trg[1:]  # p(Wn|Wn-1)
                        ug = trg[-1]  # p(Wn)
                        #if trg in trigramList[i]:
                        trg_count = trigramFreqList[i].get(' '.join(trg))
                        trg_count = int(0 if trg_count is None else trg_count)
                        bg_count = bigramFreqList[i].get(' '.join(bg))
                        bg_count = int(0 if bg_count is None else bg_count)
                        ug_count = classList[i].word_frequency.get(ug)
                        ug_count = int(0 if ug_count is None else ug_count)

                        # Interpolation lambda1*ug_prob + lambda2*bg_prob + lambda3*trg_prob
                        ug_prob = ug_count/classList[i].vocabulary_length
                        if bg_count > SMOOTHING_THRESH:
                            bg_prob = classList[i].bigram_prob(bg, bigramFreqList[i], classList[i].word_frequency)
                        else:
                            bg_prob = classList[i].good_turing(bg, bigramFreqList[i], classList[i].word_frequency,
                                                               biNCountList[i])
                        if trg_count > SMOOTHING_THRESH:
                            trg_prob = classList[i].trigram_prob(trg, trigramFreqList[i], bigramFreqList[i])
                        else:
                            trg_prob = classList[i].good_turing(trg, trigramFreqList[i], bigramFreqList[i],
                                                                triNCountList[i])
                        prob += float(lambda1 * ug_prob + lambda2 * bg_prob + lambda3 * trg_prob)
                        # else:
                            # prob += classList[i].good_turing(trg, trigramFreqList[i], bigramFreqList[i],
                                    # triNCountList[i])

                    log_prob = prob
                    prob_per_sent.append(log_prob)
                # print("predict calculation...")
                predict = prob_per_sent.index(max(prob_per_sent))
                predict_name = authorList_train[predict]
                predict_name = re.sub(r'(.*)_utf8_(.*).txt', r'\1', predict_name)

                if cor_predict_name == predict_name:
                    dev_set_predictions["C"] += 1
                else:
                    dev_set_predictions["N"] += 1
            print(dev_set_predictions)
            try:
                cor_percent = float(
                    (dev_set_predictions["C"] / (dev_set_predictions["C"] + dev_set_predictions["N"])) * 100)
            except ZeroDivisionError:
                cor_percent = 100  # if there was no wrong prediction!!
            print(f'for author {cor_predict_name}, predicted correctly {dev_set_predictions["C"]} / '
                  f'{dev_set_predictions["C"] + dev_set_predictions["N"]} = {cor_percent}%')



def classifier_test_flag(author_file, test_file):

    if TRIGRAM == False:
        print("training... (This may take a while)")
        authorList = []
        classList = []
        bigramList = []
        bigramFreqList = []
        nCountList = []

        with open(author_file, 'r', encoding="UTF-8") as fp:
            for count, line in enumerate(fp):
                s = line.rstrip('\n')
                authorList.append(s) # my file names


        for a in range(count+1):
            print(f'loading {authorList[a]}...')
            f = open("sources/"+authorList[a], 'r', encoding="UTF-8")
            classList.append(Ngram_Model(f))
            print(f'loading {authorList[a]} bigram...')
            bigramList.append(ngrams(classList[a].vocabulary, classList[a].sentences, 2))
            print(f'loading {authorList[a]} bigram frequency...')
            bigramFreqList.append(classList[a].bigram_freq(bigramList[a]))
            print(f'loading {authorList[a]} Ncounts... this takes longer than others')
            nCountList.append(classList[a].N_counts(bigramFreqList[a]))

        testfile = open(test_file, 'r', encoding="UTF-8")
        for line in testfile: # let's assume each line is a sentence!
            sentence = [filter_simple_sent(line)]
            list_of_probs = []
            for i in range(count+1):
                bigrams = ngrams(classList[i].vocabulary, list(sentence), 2)
                prob = 0
                for bg in bigrams:  # check if it's in class's bigramList
                    if bg in bigramList[i]:
                        bg_count = bigramFreqList[i].get(' '.join(bg))
                        bg_count = int(0 if bg_count is None else bg_count)
                        if bg_count > SMOOTHING_THRESH:
                            # regular count
                            prob += classList[i].bigram_prob(bg, bigramFreqList[i], classList[i].word_frequency)
                        if bg_count <= SMOOTHING_THRESH:
                            # good-turing smoothing
                            prob += classList[i].good_turing(bg, bigramFreqList[i], classList[i].word_frequency,
                                                             nCountList[i])
                    else:
                        prob += classList[i].good_turing(bg, bigramFreqList[i], classList[i].word_frequency, nCountList[i])
                log_prob = prob
                list_of_probs.append(log_prob)

            author = list_of_probs.index(max(list_of_probs))
            author_name = authorList[author]
            author_name = re.sub(r'(.*)_utf8\.txt', r'\1', author_name)  # sources?/name_utfs?.txt
            print(author_name)

    elif TRIGRAM == True:
        print("training... (This may take a while)")
        authorList = []
        classList = []
        bigramList = []
        bigramFreqList = []
        biNCountList = []
        trigramList = []
        trigramFreqList = []
        triNCountList = []

        with open(author_file, 'r', encoding="UTF-8") as fp:
            for count, line in enumerate(fp):
                s = line.rstrip('\n')
                authorList.append(s)  # my file names

        for a in range(count + 1):
            print(f'loading {authorList[a]}...')
            f = open("sources/" + authorList[a], 'r', encoding="UTF-8")
            classList.append(Ngram_Model(f))
            print(f'loading {authorList[a]} bigram...')
            bigramList.append(ngrams(classList[a].vocabulary, classList[a].sentences, 2))
            print(f'loading {authorList[a]} bigram frequency...')
            bigramFreqList.append(classList[a].bigram_freq(bigramList[a]))
            print(f'loading {authorList[a]} biNcounts... this takes longer than others')
            biNCountList.append(classList[a].N_counts(bigramFreqList[a]))
            print(f'loading {authorList[a]} trigram...')
            trigramList.append(ngrams(classList[a].vocabulary, classList[a].sentences, 3))
            print(f'loading {authorList[a]} trigram frequency...')
            trigramFreqList.append(classList[a].trigram_freq(trigramList[a]))
            print(f'loading {authorList[a]} trNcounts... this takes longer than others')
            triNCountList.append(classList[a].N_counts(trigramFreqList[a]))

        testfile = open(test_file, 'r', encoding="UTF-8")
        for line in testfile:  # let's assume each line is a sentence!
            sentence = [filter_simple_sent(line)]
            list_of_probs = []
            for i in range(count + 1):
                trigrams = ngrams(classList[i].vocabulary, list(sentence), 3)
                prob = 0
                for trg in trigrams:  # check if it's in class's bigramList
                    bg = trg[1:] # p(Wn|Wn-1)
                    ug = trg[-1] # p(Wn)
                    #if trg in trigramList[i]:
                    trg_count = trigramFreqList[i].get(' '.join(trg))
                    trg_count = int(0 if trg_count is None else trg_count)
                    bg_count = bigramFreqList[i].get(' '.join(bg))
                    bg_count = int(0 if bg_count is None else bg_count)
                    ug_count = classList[i].word_frequency.get(ug)
                    ug_count = int(0 if ug_count is None else ug_count)
                    # Interpolation lambda1*ug_prob + lambda2*bg_prob + lambda3*trg_prob
                    ug_prob = ug_count/classList[i].vocabulary_length
                    if bg_count > SMOOTHING_THRESH:
                        bg_prob = classList[i].bigram_prob(bg, bigramFreqList[i], classList[i].word_frequency)
                    else:
                        bg_prob = classList[i].good_turing(bg, bigramFreqList[i], classList[i].word_frequency,
                                                         biNCountList[i])
                    if trg_count > SMOOTHING_THRESH:
                        trg_prob = classList[i].trigram_prob(trg, trigramFreqList[i], bigramFreqList[i])
                    else:
                        trg_prob = classList[i].good_turing(trg, trigramFreqList[i], bigramFreqList[i],
                                                         triNCountList[i])
                    prob += lambda1*ug_prob + lambda2*bg_prob + lambda3*trg_prob
                    #else:
                       #prob += classList[i].good_turing(trg, trigramFreqList[i], bigramFreqList[i],
                                                         #triNCountList[i])
                log_prob = prob # division by zero for log
                list_of_probs.append(log_prob)

            author = list_of_probs.index(max(list_of_probs))
            author_name = authorList[author]
            author_name = re.sub(r'(.*)_utf8\.txt', r'\1', author_name)  # sources?/name_utfs?.txt
            print(author_name)



if __name__ == '__main__':

    #Two differet commands:
    #python3 classifier.py authorlist
    #python3 classifier.pt authorlist -test testfile
    if len(sys. argv) == 2:
        author_list = sys.argv[1]
        classifier(author_list)
    elif len(sys. argv) > 2:
        if sys.argv[2] == "-test":
            author_list = sys.argv[1]
            test_file = sys.argv[3]
            classifier_test_flag(author_list, test_file)

    #author_list = "authorlist.txt"
    #classifier(author_list)
    #test_file = "austen_test_sents copy.txt"
    #classifier_test_flag(author_list, test_file) # line by line prediction
