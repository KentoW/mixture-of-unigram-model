# -*- coding: utf-8 -*-
# 混合ユニグラムモデル(mixture of unigram model)
import sys
import math
import random
import argparse
import scipy.special
from collections import defaultdict

class MUM:
    def __init__(self, data):
        self.corpus_file = data
        self.target_word = defaultdict(int)
        self.corpus = []
        comment = ""
        for strm in open(data, "r"):
            document = {}
            if strm.startswith("#"):
                comment = strm.strip()
            else:
                if comment:
                    document["comment"] = comment
                words = strm.strip().split(" ")
                document["bag_of_words"] = words
                for v in words:
                    self.target_word[v] += 1
                self.corpus.append(document)
        self.V = float(len(self.target_word))
        # トピック分布
        self.topic_document_freq = defaultdict(float)
        self.topic_document_sum = 0.0
        # 単語分布
        self.topic_word_freq = defaultdict(lambda: defaultdict(float))
        self.topic_word_sum = defaultdict(float)

    def set_param(self, alpha, beta, K, N, converge):
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.N = N
        self.converge = converge

    def learn(self):
        self.initialize()
        self.lkhds = []
        for i in xrange(self.N):
            self.sample_corpus()
            sys.stderr.write("iteration=%d/%d K=%s alpha=%s beta=%s\n"%(i+1, self.N, self.K, self.alpha, self.beta))
            if i % 10 == 0:
                self.n = i+1
                self.lkhds.append(self.likelihood())
                sys.stderr.write("%s : likelihood=%f\n"%(i+1, self.lkhds[-1]))
                if len(self.lkhds) > 1:
                    diff = self.lkhds[-1] - self.lkhds[-2]
                    if math.fabs(diff) < self.converge:
                        break
        self.n = i+1

    def initialize(self):
        for doc in self.corpus:
            doc["state"] = 0

    def likelihood(self):
        likelihoods = []
        for doc in self.corpus:
            over_flow = []
            for z in xrange(1, self.K+1):
                l_topic = math.log((self.topic_document_freq[z] + self.alpha)/(self.topic_document_sum + (self.alpha * self.K)))
                l_words = 0.0
                for v in doc["bag_of_words"]:
                    theta = math.log((self.topic_word_freq[z][v] + self.beta)/(self.topic_word_sum[z] + (self.beta * self.V)))
                    l_words += theta
                l_doc = l_topic + l_words
                over_flow.append(l_doc)
            max_log = max(over_flow)        # オーバーフロー対策
            likelihood = 0.0
            for l_doc in over_flow:
                likelihood += math.exp(l_doc - max_log)
            likelihood = math.log(likelihood) + max_log
            likelihoods.append(likelihood)
        return sum(likelihoods)/len(likelihoods)

    def sample_corpus(self):
        for m, document in enumerate(self.corpus):
            self.sample_document(m)     # コーパス中のm番目の文書のトピックをサンプリング
        # ハイパーパラメータalphaの更新
        numerator = 0.0
        denominator = 0.0
        for z in xrange(1, self.K + 1):
            numerator += (scipy.special.digamma(self.topic_document_freq[z] + self.alpha))
        numerator -= (self.K * scipy.special.digamma(self.alpha))
        denominator = ((self.K * scipy.special.digamma(self.topic_document_sum + (self.alpha * self.K))) - (self.K * scipy.special.digamma(self.alpha * self.K)))
        self.alpha = self.alpha * (numerator / denominator)
        # ハイパーパラメータbetaの更新
        numerator = 0.0
        denominator = 0.0
        for z in xrange(1, self.K + 1):
            for v in self.target_word.iterkeys():
                numerator += scipy.special.digamma(self.topic_word_freq[z][v] + self.beta)
            denominator += scipy.special.digamma(self.topic_word_sum[z] + (self.beta * self.V))
        numerator -= (self.K * self.V * scipy.special.digamma(self.beta))
        denominator = ((self.V * denominator) - (self.K * self.V * scipy.special.digamma(self.beta * self.V)))
        self.beta = self.beta * (numerator / denominator)

    def sample_document(self, m):
        z = self.corpus[m]["state"]         # Step1: カウントを減らす
        if z > 0:
            self.topic_document_freq[z] -= 1
            self.topic_document_sum -= 1
            for v in self.corpus[m]["bag_of_words"]:
                self.topic_word_freq[z][v] -= 1
                self.topic_word_sum[z] -= 1
        n_d_v = defaultdict(float)          # Step2: 事後分布の計算
        n_d = 0.0
        for v in self.corpus[m]["bag_of_words"]:
            n_d_v[v] += 1.0
            n_d += 1.0
        p_z = defaultdict(lambda: 0.0)
        for z in xrange(1, self.K + 1):
            p_z[z] = math.log((self.topic_document_freq[z] + self.alpha) / (self.topic_document_sum + self.alpha*self.K))
            p_z[z] += (math.lgamma(self.topic_word_sum[z] + self.beta*self.V) - math.lgamma(self.topic_word_sum[z] + n_d + self.beta*self.V))
            for v in n_d_v.iterkeys():
                p_z[z] += (math.lgamma(self.topic_word_freq[z][v] + n_d_v[v] + self.beta) - math.lgamma(self.topic_word_freq[z][v] + self.beta))
        max_log = max(p_z.values())     # オーバーフロー対策
        for z in p_z:
            p_z[z] = math.exp(p_z[z] - max_log)
        new_z = self.sample_one(p_z)        # Step3: サンプル
        self.corpus[m]["state"] = new_z     # Step4: カウントを増やす
        self.topic_document_freq[new_z] += 1
        self.topic_document_sum += 1
        for v in self.corpus[m]["bag_of_words"]:
            self.topic_word_freq[new_z][v] += 1
            self.topic_word_sum[new_z] += 1

    def sample_one(self, prob_dict):
        z = sum(prob_dict.values())                     # 確率の和を計算
        remaining = random.uniform(0, z)                # [0, z)の一様分布に従って乱数を生成
        for state, prob in prob_dict.iteritems():       # 可能な確率を全て考慮(状態数でイテレーション)
            remaining -= prob                           # 現在の仮説の確率を引く
            if remaining < 0.0:                         # ゼロより小さくなったなら，サンプルのIDを返す
                return state

    def output_model(self):
        print "model\tmixture_of_unigram_model"
        print "@parameter"
        print "corpus_file\t%s"%self.corpus_file
        print "hyper_parameter_alpha\t%f"%self.alpha
        print "hyper_parameter_beta\t%f"%self.beta
        print "number_of_topic\t%d"%self.K
        print "number_of_iteration\t%d"%self.n
        print "@likelihood"
        print "initial likelihood\t%s"%(self.lkhds[0])
        print "last likelihood\t%s"%(self.lkhds[-1])
        print "@vocaburary"
        for v in self.target_word:
            print "target_word\t%s"%v
        print "@count"
        for z, freq in self.topic_document_freq.iteritems():
            print 'topic_document_freq\t%s\t%d' % (z, freq)
        for z, word_freq_dict in self.topic_word_freq.iteritems():
            print 'topic_word_sum\t%s\t%d' % (z, self.topic_word_sum[z])
            for v, freq in sorted(word_freq_dict.iteritems(), key=lambda x:x[1], reverse=True):
                if int(freq) != 0:
                    print 'topic_word_freq\t%s\t%s\t%d' % (z, v, freq)
        print "@data"
        for document in self.corpus:
            if "comment" in document:
                print "# state", document["state"], document["comment"]
            else:
                print "#state %d"%(document["state"])
            print " ".join(document["bag_of_words"])


def main(args):
    mum = MUM(args.data)
    mum.set_param(args.alpha, args.beta, args.K, args.N, args.converge)
    mum.learn()
    mum.output_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alpha", dest="alpha", default=0.01, type=float, help="hyper parameter alpha")
    parser.add_argument("-b", "--beta", dest="beta", default=0.01, type=float, help="hyper parameter beta")
    parser.add_argument("-k", "--K", dest="K", default=10, type=int, help="topic")
    parser.add_argument("-n", "--N", dest="N", default=1000, type=int, help="max iteration")
    parser.add_argument("-c", "--converge", dest="converge", default=0.01, type=str, help="converge")
    parser.add_argument("-d", "--data", dest="data", default="data.txt", type=str, help="training data")
    args = parser.parse_args()
    main(args)
