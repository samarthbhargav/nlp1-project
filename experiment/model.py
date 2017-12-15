import numpy as np
import argparse
import codecs
import math
import os
import sys
from itertools import count, takewhile, zip_longest
from collections import defaultdict

import utils

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.abspath("../"))

from opennmt import opts
from opennmt.onmt import IO
from opennmt import onmt
from data import *

nlp_en_notok = spacy.load("en", disable=["tokenizer"])
nlp_nl_notok = spacy.load("nl", disable=["tokenizer"])



def report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total / words_total)))


def get_src_words(src_indices, index2str):
    words = []
    raw_words = (index2str[i] for i in src_indices)
    words = takewhile(lambda w: w != onmt.IO.PAD_WORD, raw_words)
    return " ".join(words)


def construct_args(model_path):
    args = ["-model", model_path, "-src", "europalProcessedEN.txt"]
    sys.argv = sys.argv[:] + args


def plot_attention(path, source_words, target_words, attention):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)


    attention = attention[0]


    attention = attention.numpy()
    cax = ax.matshow(attention[:, :len(source_words) + 1], cmap='RdPu')
    fig.colorbar(cax, ticks=[0, 1]  )

    en_orig_tokens = " ".join(source_words)
    en_tokens = nlp_en(spacy_fix(en_orig_tokens))
    en_token_pos = []
    for tok in en_tokens:
        en_token_pos.append(tok.pos_ + " " + tok.text)

    nl_orig_tokens = " ".join(target_words)
    nl_tokens = nlp_nl(spacy_fix(nl_orig_tokens))
    nl_token_pos = []
    for tok in nl_tokens:
        nl_token_pos.append(tok.pos_ + " " + tok.text)


    ax.set_xticklabels([''] + en_token_pos + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + nl_token_pos )

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(path, dpi=600, bbox_inches='tight')

def spacy_fix(sentence):
        return sentence.replace("<unk>", "UNK").replace("&apos;", "'")


class Scorer:
    def __init__(self, k=None):
        if k is None:
            k = [3]
        
        # k must be odd
        assert all(_ % 2 != 0 for _ in k)
        
        self.k = k
        
        # total number of sentences
        self.total = 0
        
        # number of perfect verb alignments
        self.perfect_verb_match_pos = 0 
        self.perfect_verb_match_dep = 0
        
        # number of top k verb matches
        self.top_k_verb_match_dep = defaultdict(int)
        self.top_k_verb_match_pos = defaultdict(int)

        # confusion matrix for perfect verb alignments
        # structure is POS - count
        self.conf_matrix_dep = defaultdict(int) 
        self.conf_matrix_pos = defaultdict(int)

    def _find_verb_nl_pos(self, doc):
        """ 
        Finds and returns the position of the verb in the sentence
        """
        for index, token in reversed(list(enumerate(doc))):
            if token.pos_ == "VERB":
                return index
    
    def _find_verb_nl_dep(self, doc):
        for index, token in reversed(list(enumerate(doc))):
            if token.dep_ == "ROOT":
                return index

    def _find_verb_en(self, doc):
        """
        Finds and returns the position of the first verb in the sentence
        """
        for index, token in list(enumerate(doc)):
            if token.dep_ == "ROOT":
                return index

    def spacy_fix(self, sentence):
        return sentence.replace("<unk>", "UNK").replace("&apos;", "'")

    def accumulate_scores(self, source_sentence, target_sentence, true_sentence,
            attention, jaccard_threshold=0.3):
        attention = attention[0]
        
        # fix so that spacy doesn't tokenize <unk> into [< , UNK, >]
        source_sentence = self.spacy_fix(source_sentence)
        target_sentence = self.spacy_fix(target_sentence)
        true_sentence = self.spacy_fix(true_sentence)

        en_doc = nlp_en(source_sentence)
        en_doc = [tok for tok in en_doc]
        nl_doc = nlp_nl(target_sentence)
        nl_doc = [tok for tok in nl_doc]
        true_nl_doc = nlp_nl(true_sentence)
        true_nl_doc = [tok for tok in true_nl_doc]

        print("\tSentence: {}\n\tTranslation:{}\n\tActual Translation:{}".format([(tok.text, tok.pos_,
            tok.dep_) for tok in en_doc],[(tok.text, tok.pos_, tok.dep_) for tok in nl_doc], 
            [(tok.text, tok.pos_, tok.dep_) for tok in true_nl_doc]))

        jaccard_index = utils.jaccard_index([tok.text for tok in nl_doc],
                [tok.text for tok in true_nl_doc])

        print("\tJaccard Index: {}".format(jaccard_index))

        if jaccard_index < jaccard_threshold:
            print("\tJaccard Index not high enough. Not considering this sentence for scoring")
            return 

        verb_index_en = self._find_verb_en(en_doc)
        
        print("\t*** POS ***")
        # find the ending verb, using POS first
        verb_index_nl = self._find_verb_nl_pos(nl_doc)
        

        print("\tSource Verb: {}\n\tTarget Verb:{}".format(en_doc[verb_index_en],
            nl_doc[verb_index_nl]))

        verb_attention = attention[:, verb_index_nl].numpy()
         
        pred_max_attention = verb_attention.argmax()
        if len(en_doc) < pred_max_attention:
            print("\tIncorrect Dimensions :(")
            return

        print("\tEnglish Index: {}, NL Actual Index: {}, Predicted Index:{}\n\tAttention: {}".format(verb_index_en, 
            verb_index_nl, pred_max_attention, verb_attention))

        # check for perfect match
        if pred_max_attention == verb_index_en:
            print("\tPerfect Match!")
            self.perfect_verb_match_pos += 1
        else:
            print("\tNot a perfect match. Instead: {}".format(en_doc[pred_max_attention]))
            
            # update confusion matrix
            self.conf_matrix_pos[en_doc[pred_max_attention].pos_] += 1

            for k in self.k:
                print("\tFor window: {}".format(k))
                # check if it's a k/2 sized window
                start = max(0, pred_max_attention - (k-1)//2)
                end = min(len(en_doc), pred_max_attention + (k-1)//2 + 1)
                allowed_range = np.arange(start, end)
                print("\t\tAllowed Range: {}".format(allowed_range))
                if verb_index_en in allowed_range:
                    print("\t\tIt's in the allowed range")
                    self.top_k_verb_match_pos[k] += 1
                else:
                    print("\t\tIt's not in the allowed range :(")
        
        verb_index_nl = self._find_verb_nl_dep(nl_doc)

        print("\t*** DEP ***")
        # find the ending verb, using POS first
        verb_index_nl = self._find_verb_nl_dep(nl_doc)
        

        print("\tSource Verb: {}\n\tTarget Verb:{}".format(en_doc[verb_index_en],
            nl_doc[verb_index_nl]))

        verb_attention = attention[:, verb_index_nl].numpy()
         
        pred_max_attention = verb_attention.argmax()
        print("\tEnglish Index: {}, NL Actual Index: {}, Predicted Index:{}\n\tAttention: {}".format(verb_index_en, 
            verb_index_nl, pred_max_attention, verb_attention))

        # check for perfect match
        if pred_max_attention == verb_index_en:
            print("\tPerfect Match!")
            self.perfect_verb_match_dep += 1
        else:
            print("\tNot a perfect match. Instead: {}".format(en_doc[pred_max_attention]))
            
            # update confusion matrix
            self.conf_matrix_dep[en_doc[pred_max_attention].pos_] += 1

            for k in self.k:
                print("\tFor window: {}".format(k))
                # check if it's a k/2 sized window
                start = max(0, pred_max_attention - (k-1)//2)
                end = min(len(en_doc), pred_max_attention + (k-1)//2 + 1)
                allowed_range = np.arange(start, end)
                print("\t\tAllowed Range: {}".format(allowed_range))

                if verb_index_en in allowed_range:
                    print("\t\tIt's in the allowed range")
                    self.top_k_verb_match_dep[k] += 1
                else:
                    print("\t\tIt's not in the allowed range :(")
        print("\t**********************")
        
        self.total += 1


def read_true_nl():
    true_nl_sentences = []
    with codecs.open("europalProcessedNL.txt", "r", "utf-8") as reader:
        for line in reader:
            true_nl_sentences.append(line)
    return true_nl_sentences

if __name__ == '__main__':
    ### PARAMs for Model
    model = "../models/ted_sgd_acc_55.43_ppl_12.39_e11.pt"
    ##
    construct_args(model)

    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.translate_opts(parser)
    opt = parser.parse_args()

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    opt.cuda = opt.gpu > -1

    dummy_opt = dummy_parser.parse_known_args([])[0]

    translator = onmt.Translator(opt, dummy_opt.__dict__)

    data = IO.ONMTDataset(
        opt.src, opt.tgt, translator.fields,
        use_filter_pred=False)

    test_data = IO.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        shuffle=False)
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    out_file = codecs.open(opt.output, 'w', 'utf-8')

    counter = count(1)
    sentence = 0

    scorer = Scorer([3, 5, 7])

    true_nl_sentences = read_true_nl()

    for batch in test_data:
        pred_batch, gold_batch, pred_scores, gold_scores, attn, src \
            = translator.translate(batch, data)
        pred_score_total += sum(score[0] for score in pred_scores)
        pred_words_total += sum(len(x[0]) for x in pred_batch)
        if opt.tgt:
            gold_score_total += sum(gold_scores)
            gold_words_total += sum(len(x) for x in batch.tgt[1:])

        # z_batch: an iterator over the predictions, their scores,
        # the gold sentence, its score, and the source sentence for each
        # sentence in the batch. It has to be zip_longest instead of
        # plain-old zip because the gold_batch has length 0 if the target
        # is not included.
        z_batch = zip_longest(
            pred_batch, gold_batch,
            pred_scores, gold_scores,
            (sent.squeeze(1) for sent in src.split(1, dim=1)))

        for index, (pred_sents, gold_sent, pred_score, gold_score, src_sent) in enumerate(z_batch):
            n_best_preds = [" ".join(pred) for pred in pred_sents[:opt.n_best]]
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()

            sent_number = next(counter)
            words = get_src_words(
                src_sent, translator.fields["src"].vocab.itos)

            os.write(1, bytes('\nSENT %d: %s\n' %
                              (sent_number, words), 'UTF-8'))

            best_pred = n_best_preds[0]
            best_score = pred_score[0]
            os.write(1, bytes('PRED %d: %s\n' %
                              (sent_number, best_pred), 'UTF-8'))
            print("PRED SCORE: %.4f" % best_score)

            plot_attention("attentions/{}.png".format(sentence + 1),
                           words.split(), best_pred.split(), attn[index])
            try:
                scorer.accumulate_scores(words, best_pred,
                        true_nl_sentences[sentence], attn[index])
            except:
                ...
            print("\n\n\n")


            sentence += 1
        if sentence > 500:
            break

    report_score('PRED', pred_score_total, pred_words_total)
    
    import json
    print(json.dumps(scorer.__dict__, indent=2))

    with open("results.json", "w") as writer:
        writer.write(json.dumps(scorer.__dict__, indent=2))
