import sys
import re
import numpy as np

DELETION_COST = 1
INSERTION_COST = 1
SUBSTITUTION_COST = 2

DISTANCE_THRESHOLD = 5

# for tokenization
CLITICS = ["I've", "I'm", "I'd", "I'll",
           "she'll", "she's", "she'd", "She'll", "She's", "She'd",
           "he'll", "he's", "he'd", "He'll", "He's", "He'd",
           "they'll", "they've", "they'd", "they're", "They'll", "They've", "They'd", "They're",
           "we'll", "we've", "we'd", "We'll", "We've", "We'd",
           "you're", "you've", "you'd", "y'all", "You're", "You've", "You'd", "Y'all",
           "it's", "it'll", "It's", "It'll",

           ]
PERIODS = ["Mr.", "Mrs.", "Ms.", "Dr.", "Jr.", "Sr.", "e.g.", "i.e.", "Prof.", "etc.", "a.m.", "p.m."]

# disctionary not be in lower if in this list:
DATE = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"]
CAPS = ["Mr", "Mrs", "Ms", "Dr", "Jr", "Sr",
        "USA", "UK", "NASA", "NATO", "NAFTA", "EU", "UN", "UNICEF", "WHO", "CDC", "FBI", "CIA", "CPU",
        "MD", "CEO", "HTML", "DNA", "ATM", "URL", "PC", "PhD", "MBA", "BSc", "MSc", "FYI"
        ]
UNITS = ["km", "ms", "cm", "ml", "rpm", "mph", "mps", "mph", "mW", "emf", "Gb", "sq", "bps",
         "dpi", "bhp", "dc", "fp", "bps", "ppm", "www", "com"]


def tokenization(line):
    tokens_list = []

    line = re.sub(r'[\"!?;:(){}\[\]\\#]', r'', line)  # no comma, period (only if its not numbers or URL)
    # whitespace around commas that aren't in numbers
    line = re.sub(r'([^0-9]),', r'\1 , ', line)  # add space between word and ,
    line = re.sub(r',([^0-9])', r' , \1', line)  # add space if , before word
    line = re.sub(r'([0-9]*),', r'\1 , ', line)  # add space between number and .
    line = re.sub(r'([0-9]*)\.', r'\1 \. ', line)  # add space between number and ,
    line = re.sub(r'([0-9]*) ', r' ', line)  # replace numbers with nothing
    line = re.sub(r' , ', r' ', line)  # remove commas if whitespace around it
    line = re.sub(r' \\. ', r' ', line)  # remove periods if whitespace around it


    split_list = line.split()
    for token in split_list:
        if token not in CLITICS:
            token = re.sub(r"'", r'', token)
        if token not in PERIODS:
            token = re.sub(r'\\.', r'', token)
        tokens_list.append(token)

    return tokens_list


def edit_dist(target, source):
    # Levenshtein version of distance costs, insertion=deletion=1 substitution=2
    n = len(target)
    m = len(source)
    D = np.zeros((n + 1, m + 1))

    for i in range(1, n + 1):
        D[i, 0] = D[i - 1, 0] + INSERTION_COST
    for j in range(1, m + 1):
        D[0, j] = D[0, j - 1] + DELETION_COST

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if target[i - 1] == source[j - 1]:
                sub = 0
            else:
                sub = SUBSTITUTION_COST
            D[i, j] = min(D[i - 1, j] + 1, D[i - 1, j - 1] + sub, D[i, j - 1] + 1)

    return int(D[n, m])


def dict_check(word):

    if word.isupper():  # if word is all in caps, just accept (maybe abbreviation/name of something)
        return True
    elif word in CAPS:
        return True
    elif word in UNITS:
        return True
    elif word in CLITICS:
        return True
    elif word in PERIODS:
        return True

    fdict = open("words")
    for line in fdict:
        dict_word = line.split()[0]
        if dict_word not in DATE or CAPS:
            if word.lower() == dict_word.lower():
                return True
        else:
            if word == dict_word:
                return True
    return False


def suggest(incorrect_word):
    # given an incorrectly-spelled word,return a list of at most 3 words with the least edit distance
    # shortcuts for shorter search: Edit distance threshold + word length

    dist_dict = {}
    inc_word_len = len(incorrect_word)
    fdict = open("words")
    for line in fdict:
        word = line.split()[0]
        word_len = len(word)
        if inc_word_len - 2 < word_len < inc_word_len + 2:
            dist = edit_dist(incorrect_word, word)

            if dist < DISTANCE_THRESHOLD:
                if word not in dist_dict.keys():
                    dist_dict.update({word: dist})

    # sorting the dictionary based on its values
    sorted_dict = sorted(dist_dict.items(), key=lambda x: x[1], reverse=False)

    # saving the keys in the list
    dist_list = [k for k, v in sorted_dict]

    return dist_list[0:3]  # up to 3 words


def main():
    # 1.read in text file as a command line argument
    # 2.look for words that are incorrectly spelled.
    # 3.finds a word that is spelled incorrectly,print out the line that the word appeared in,
    # and a list of possible suggestions for correcting the word
    # for now just choose the first word correction and move on....
    # 4.go through entire document
    # 5.save its corrected version to a new file with the prefix `corrected_`
    # 6.For fun, you may add in an interactive interface that asks the user which of the three choices that they wanted

    ''' My approach:
    open the text file
    read first line
    tokenize it
    check if there is an incorrectly spelled word
    then find suggestions and print it out
    fix the word in the line and write it into a new file
    go to the next line and repeat the process
    '''

    f_inc = open(sys.argv[1])  # incorrect file
    ''' getting the file name and creating a new corrected file'''
    obj = re.search(r"(.*).txt", f_inc.name)
    corr_fname = re.sub(obj.group(1), f'corrected_{obj.group(1)}', f_inc.name)
    f_c = open(corr_fname, "w")

    corr_dict = {}

    print()
    print("Hello my friend!")
    print("Would you like to use the interactive interface to select from suggestion list? y/n")
    interface_s = input()
    if interface_s == "y":
        INTERACTION = True
        print("Great! Let's start then :)")
    elif interface_s == "n":
        print("OK then I will always pick the first word then.")
        INTERACTION = False
    else:
        print("Sorry I didn't get it. I will move on without interaction.")
        INTERACTION = False

    for line in f_inc:
        # 1. Toeknization
        line_tok = tokenization(line)
        # 2. dictionary check
        for item in line_tok:
            if dict_check(item) is False:
                # 3. get the suggestions and print them out
                sugg_list = suggest(item)

                print()
                print("***************************************************")
                print(f'"{item}" is spelled wrong in the following line:')
                print(f'"{line[:-1]}"')  # bc line includes \n at the end!
                print("Now here are my suggestions for correcting the word: ")
                print(sugg_list)
                print()

                if INTERACTION:
                    print("Please select one of the options from the suggestion list using 1-3")
                    select = int(input()) - 1
                    # make a dictionary of incorrect and correct word
                    corr_dict[item] = sugg_list[select]
                else:
                    corr_dict[item] = sugg_list[0]

                print("***************************************************")
                print()

        # 4. line is over and write it into the new file with correct words
        corr_line = line[:]
        for inc, c in corr_dict.items():
            corr_line = line.replace(inc, c)
        f_c.write(corr_line)

    f_c.close()
    f_inc.close()


if __name__ == '__main__':
    main()
