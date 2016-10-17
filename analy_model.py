# -*- coding: utf-8 -*-
import sys
import collections
import argparse

def main(args):
    max_word_state = collections.defaultdict(int)
    max_word_prob = collections.defaultdict(float)
    for strm in open(args.model):
        if strm.startswith("topic_word_sum"):
            #print "state:", strm.strip().split("\t")[1]
            state_word_sum = float(strm.split("\t")[2])
        if strm.startswith("topic_word_freq"):
            state = strm.strip().split("\t")[1]
            word = strm.strip().split("\t")[2]
            freq = float(strm.strip().split("\t")[3])
            if state_word_sum == 0:
                prob = 0.0
            else:
                prob = freq / state_word_sum
            #print word, prob
            if prob > max_word_prob[word]:
                max_word_state[word] = int(state)
                max_word_prob[word] = prob

    hoge = []
    i = 1
    for strm in open(args.model):
        if strm.startswith("topic_word_sum"):
            c = 0
            if hoge:
                hoge_set = []
                outhoge = []
                for w in hoge:
                    if w in hoge_set:
                        continue
                    hoge_set.append(w)
                    outhoge.append(w)
                #print str(i)+"&&"+",".join(hoge[:50:])+"\\\\ \\hline"
                print str(i), len(hoge_set), ",".join(hoge_set[:100:])
                print ""
                i += 1
            hoge = []
            #print "state:", strm.strip().split("\t")[1], strm.strip().split("\t")[2]
        if strm.startswith("topic_word_freq"):
            state = strm.strip().split("\t")[1]
            word = strm.strip().split("\t")[2]
#            sur = word.split("|")[0]
#            pos1 = word.split("|")[1]
#            pos2 = word.split("|")[2]
            freq = float(strm.strip().split("\t")[3])
            if state_word_sum == 0:
                prob = 0.0
            else:
                prob = freq / state_word_sum
            if max_word_state[word] == int(state):
                #if (pos1 == "名詞" and pos2 == "一般") or (pos1 == "動詞" and pos2 == "自立") or (pos1 == "形容詞" and pos2 == "自立") or (pos1 == "助詞") or (pos1 == "助動詞") or sur == "ukn":
                    #if c < 20:
                #print "%s\t%s"%(word.split("|")[0], prob)
                hoge.append(word.split("|")[0])
                c += 1
    if hoge:
        hoge_set = []
        outhoge = []
        for w in hoge:
            if w in hoge_set:
                continue
            hoge_set.append(w)
            outhoge.append(w)
        #print str(i)+"&&"+",".join(hoge[:50:])+"\\\\ \\hline"
        print str(i), len(hoge_set), ",".join(hoge_set[:100:])
        print ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", default="./6to18/model_k20_a0.01_b1_6to18.txt", type=str, help="specify model file name")
    args = parser.parse_args()
    main(args)
