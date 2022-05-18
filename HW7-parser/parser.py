import sys
import nltk
import re
import numpy as np


class Node:
    #Node is a rule in cnf,  parent is lhs
    # Two options for childs:
    #1- child1 and child2 are non-terminal symbols from rhs
    #2- child1 is terminal and child2 is none
    def __init__(self, parent, child1, child2=None):
        self.parent = parent
        self.child1 = child1
        self.child2 = child2


def read_grammar(filename):
    # basically just to avoid the comment line
    cfg_g = []
    f = open(filename, 'r')
    with open(filename, 'r') as fp:
        for line in fp:
            # if it starts with # it's a comment , go to next
            if re.search("#", line):
                pass
            else:
                a = line.split(' -> ')
                if a[0][0].isupper():
                    rhs = re.sub(r'\n', r'', a[1])
                    # rhs = list(rhs.split(' '))
                    rule = [a[0], rhs]
                    cfg_g.append(rule)
                # print(cfg_g)
        # print(cfg_g)
    return cfg_g


def cfg_to_cnf(grammar):
    # given list of cfg rules, convert it to cnf rules
    cfg_g = grammar.tolist()
    #cfg_g = g
    cnf_g = []
    i = 1
    unit_productions=[]

    # First copy all those rules that comply with CNF
    for rule in grammar:
        lhs_list = list(rule[0].split(' '))
        rhs_list = rule[1]
        if len(rhs_list) > 1:
            rhs_list = list(rule[1].split(' '))

        # 1 check if lhs is 1 and rhs is only 2 upper case A -> B C , move to cnf
        if rule[0][0].isupper() and len(rhs_list) == 2 and rhs_list[0][0].isupper() and rhs_list[1][0].isupper():
            cnf_g.append(list(rule))
            cfg_g.remove(list(rule))

        # 2 check if lhs if upper and rhs is only ONE lower A -> a , move to cnf
        elif rule[0][0].isupper() and rule[1].islower() and len(lhs_list) == 1 and len(rhs_list) == 1:
            cnf_g.append(list(rule))
            cfg_g.remove(list(rule))

        # 5 unit production do NOT eliminate, S->E look for E->e then S->e otherwise just eliminate
        elif rule[0][0].isupper() and rule[1][0].isupper() and len(lhs_list) == 1 and len(rhs_list) == 1:
            unit_productions.append(list(rule))
            #cnf_g.append(list(rule))
            cfg_g.remove(list(rule))

    while (unit_productions): # S -> VP
        rule = unit_productions.pop()
        if rule[1] in grammar[:,0]: # if there is VP -> ?
            new_g = []
            for r in grammar:
                if r[0]==rule[1] :
                    new_g.append(r)
            for e in new_g:
                new_uni_rule = [rule[0], e[1]]
                rhs = e[1].split(' ')
                if len(rhs)==1:
                    if e[1][0].islower():  # A -> a // work is done
                        # update cnf_grammar for the rule
                        cnf_g.append(new_uni_rule)
                    else:
                        unit_productions.append(new_uni_rule)
                elif len(rhs)==2: # A -> B C
                    if new_uni_rule not in cnf_g:
                        cnf_g.append(new_uni_rule)


    cnf_g = np.asarray(cnf_g)
    # now for those rules which don't comply with chomsky normal form
    for rule in cfg_g:
        lhs_list = list(rule[0].split(' '))
        rhs_list = rule[1]
        if len(rhs_list) > 1:
            rhs_list = list(rule[1].split(' '))

        if rule[0].isupper() and len(lhs_list) == 1 and len(rhs_list) == 3:
            # 3 check for A -> B C D
            if rhs_list[0][0].isupper() and rhs_list[1][0].isupper() and rhs_list[2][0].isupper():
                # check if there is anything going to B C or C D
                m = ' '.join(rhs_list[:2])
                n = ' '.join(rhs_list[1:])

                new_symbol = 'X' + str(i)  # x1 -> B C
                i += 1
                temp_rule = [new_symbol, str(' '.join(rhs_list[0:2]))]
                new_rule = [rule[0], new_symbol + ' ' + rhs_list[2]]
                cnf_g = np.vstack((cnf_g, temp_rule))
                cnf_g = np.vstack((cnf_g, new_rule))
                #cfg_g.remove(list(rule))
            # 4 check for A -> B foo D
            else:
                for s in rhs_list:
                    idx = rhs_list.index(s)
                    if s[0].islower():
                        if s in cnf_g[:,1]: #if there is any non-t symbol for it
                            line_symbol = np.where(cnf_g[:,1]==s)
                            #rhs_list[idx]= np.asscalar(cnf_g[line_symbol,0])
                            rhs_list[idx] = cnf_g[line_symbol, 0].item()
                            rule[1] = ' '.join(rhs_list)
                            #rule[1][index(s)] = cnf_g[line_symbol:,0]
                        else:
                            new_symbol = 'X' + str(i)  # x1 -> B C
                            i += 1
                            new_rule = [new_symbol, s]
                            cnf_g = np.vstack((cnf_g, new_rule))
                            rhs_list[idx] = new_symbol
                            rule[1] = ' '.join(rhs_list)

                new_symbol = 'X' + str(i)  # x1 -> B C
                i += 1
                temp_rule = [new_symbol, str(' '.join(rhs_list[0:2]))]
                new_rule = [rule[0], new_symbol + ' ' + rhs_list[2]]
                cnf_g = np.vstack((cnf_g, temp_rule))
                cnf_g = np.vstack((cnf_g, new_rule))

        # >>>>> check for repetition
    return cnf_g


def divide_rules(cnf_grammar):
    # divide grammar to two set of rules ( A -> B C , A -> a)
    rules = cnf_grammar
    nt_rules = []
    t_rules = []

    for rule in rules:
        left, right = rule[0], rule[1]
        r = right[0]
        # it is a terminal
        if r.islower():
            t_rules.append(rule)
        # it is a variable
        else:
            nt_rules.append(rule)

    return nt_rules, t_rules


def cky_recognizer(nt, t, s):
    # building cky parse tree
    n = len(s)

    # Initialize the table
    parse_table = [[[] for i in range(n)] for j in range(n)]

    for j, word in enumerate(s):
        # go through every column, from left to right
        for rule in t:
            # fill the terminal word cell
            if word == rule[1]:
                parse_table[j][j].append(Node(rule[0], word))
        # go through every row, from bottom to top
        for i in range(j - 1, -1, -1):
            for k in range(i, j):
                child1_cell = parse_table[i][k]  # cell left
                child2_cell = parse_table[k + 1][j]  # cell beneath
                for rule in nt:
                    rhs = rule[1].split(' ')
                    child1_node = [n for n in child1_cell if n.parent == rhs[0]]
                    if child1_node:
                        child2_node = [n for n in child2_cell if n.parent == rhs[1]]
                        parse_table[i][j].extend(
                            [Node(rule[0], child1, child2) for child1 in child1_node for child2 in child2_node]
                        )
    return parse_table # will decide later if tree exists or not


def draw_parse_tree(parse_table):

    start_symbol = 'S'
    # final_nodes is the the cell in the upper right hand corner of the parse_table
    # we choose the node whose parent is the start_symbol
    final_nodes = [n for n in parse_table[0][-1] if n.parent == start_symbol]
    if final_nodes:
        print("***  Parse Tree exist for the given sentence  ***")
        # print the parse tree
        print("Possible tree(s):")
        write_trees = [generate_tree(node) for node in final_nodes]
        for tree in write_trees:
            print(tree)
        # draw the parse tree
        draw_trees = [visualize_nltk_tree(node) for node in final_nodes]
        for tree in draw_trees:
            tree.draw()
    else:
        print("Parse Tree does NOT exist for the given sentence. sorry!")


def generate_tree(node):

    if node.child2 is None:
        return f"[{node.parent} '{node.child1}']"
    return f"[{node.parent} {generate_tree(node.child1)} {generate_tree(node.child2)}]"


def visualize_nltk_tree(node):

    if node.child2 is None:
        return nltk.Tree(node.parent, [node.child1])
    return nltk.Tree(node.parent, [visualize_nltk_tree(node.child1), visualize_nltk_tree(node.child2)])


if __name__ == '__main__':
    # given wav file name, display a portion of spectogram of the file
    grammar_f = sys.argv[1]
    sentence = str(sys.argv[2]).split(' ')
    #grammar_f = 'l1.cfg'
    #sentence = 'book this flight to houston'.split(' ')

    cfg_g = read_grammar(grammar_f) # read the grammar file, ignoreing comments
    cfg_g = np.asarray(cfg_g)

    cnf_g = cfg_to_cnf(cfg_g)  # giving list of cfg rules , expecting list of cnf rules
    cnf_g = cnf_g.tolist() # i convert it to np array to iterate easier

    CNF = []
    for i in cnf_g: # removing repetitions from the grammar list
        if i not in CNF:
            CNF.append(i)
    new_g = open("new_grammar.txt",'w')
    with open('new_grammar.txt', 'w') as fw:
        for r in cnf_g:
            new_g.write(str(r)+"\n")
        print("check out the new file that is the grammar in Chomsky Normal Form")
        new_g.close()

    nt_rules, t_rules = divide_rules(CNF) # divide non-terminal and terminal rules for cky algo

    table = cky_recognizer(nt_rules,  t_rules, sentence) # building the cky parse tree

    draw_parse_tree(table) # see if the tree exists and then visualize it
    print("All done! :)")