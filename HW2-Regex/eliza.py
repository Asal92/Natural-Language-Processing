import re

full_sub = {
            "yes": "I see.",
            "no": "Why not?",
            "goodbye": "Goodbye!",
            r"(.*)you$": "Let's not talk about me.",
            r"my (.*)": r"your \1",
            r"your (.*)": r"my \1",
            r"(i am) (.*)": r"Do you enjoy being \2?",
}

def fullSub(s):
    for key in full_sub.keys():
        if re.fullmatch(key, s):
            new_s = re.sub(key, full_sub[key], s)
            return new_s
    return None

group_sub = {
            r"(what is) (.*)": "Why do you ask about",
            r"(why is) (.*)": "Why do you think",
}
def groupSub(s):
    for key in group_sub.keys():
        if re.match(key,s):
            def groupCall(m):
                return f'{group_sub[key]} {m.group(2)}'
            new_s = re.sub(key,groupCall, s)
            return new_s
    return None

expansion_sub = {
            r"(.*)i'm (.+) all the time": r"why are you \2 all the time?",
            r".*all .*": r"In whay way?",
            r".*always.*": r"Can you think of a specific example?",
            r"i like (.*)": r"Why do you like \1 ?",
            r"(.*) or (.*)": r"Tell me more about \1 or \2",
            r"i want to become (a|an) \b([^ ]*)\b": r"why do you think about becoming \1 \2?",
}

def expansionSub(s):
    for key in expansion_sub.keys():
        if re.fullmatch(key, s):
            new_s = re.sub(key, expansion_sub[key], s)
            return new_s
    return None


def main():
    print("Eliza: Hello. Please tell me about your problems.")

    while True:
        sentence = input("You: ")
        sentence = sentence.lower()
        re.compile(sentence)
        if sentence == "quit":
            break

        if fullSub(sentence):
            new_sentence = fullSub(sentence)
        elif groupSub(sentence):
            new_sentence = groupSub(sentence)
        elif expansionSub(sentence):
            new_sentence = expansionSub(sentence)
        else:
            new_sentence = "Please go on."

        print("Eliza: " + new_sentence)




if __name__ == '__main__':
    main()


