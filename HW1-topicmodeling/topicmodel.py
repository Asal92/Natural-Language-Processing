import sys
from collections import Counter

delete_words = ["a", "an", "about", "above", "across", "after", "against", "along", "among", "and", "around",
                "at", "before", "behind", "between", "beyond", "but", "by", "concerning", "so", "was", "here"
                                                                                                       "despite",
                "down", "during", "except", "following", "for", "from", "in", "if", "yes", "no",
                "including", "into", "is", "like", "near", "of", "off", "on", "or", "onto", "out",
                "over", "past", "plus", "since", "that", "the", "throughout", "to", "towards", "under",
                "until", "up", "upon", "up to", "with", "within", "without", "have", "not", "be",
                "which", "what", "who", "when", "i", "are", "as", "we", "me", "she", "he", "our", "it", "do", "must",
                "will", "those", "they", "has", "this", "other", "been", "their", "her", "his", "such", "any", "only",
                "can", "could", "should", "shall", "would", "all", "its", "made", "more", "some", "being", "there"]


def best_words(f):
    words = []  # for saving outputs
    lines = f.readlines()
    newLines = []
    for line in lines:
        line = line.lower()
        for word in line.split():
            if word not in delete_words:
                newLines.append(word)

    cnts = {}
    for word in newLines:
        cnts[word] = newLines.count(word)

    c = Counter(cnts)
    maxvals = c.most_common(5)  # returns top 5 keys with max values
    words = [key for key, val in maxvals]

    return words


def main():
    f = open(sys.argv[1])
    words = best_words(f)
    for word in words:
        print(word)
    f.close()


if __name__ == '__main__':
    main()
