import numpy as np
import argparse
import codecs
import math
import os
import sys
from itertools import count, takewhile, zip_longest
from collections import defaultdict

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
    cax = ax.matshow(attention[:, :len(source_words) + 1], cmap='bone')
    fig.colorbar(cax, ticks=[0, 1])
    

    en_orig_tokens = " ".join(source_words)
    en_tokens = nlp_en(en_orig_tokens)
    en_token_pos = []
    for tok in en_tokens:
        en_token_pos.append(tok.pos_ + " " + tok.text)

    nl_orig_tokens = " ".join(target_words)
    nl_tokens = nlp_nl(nl_orig_tokens)
    nl_token_pos = []
    for tok in nl_tokens:
        nl_token_pos.append(tok.pos_ + " " + tok.text)


    ax.set_xticklabels([''] + en_token_pos + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + nl_token_pos )

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(path, dpi=600, bbox_inches='tight')

class Scorer:
    def __init__(self, k=3):
        # k must be odd
        assert k % 2 == 1

        # total number of sentences
        self.total = 0
        # number of perfect verb alignments
        self.perfect_verb_match = 0 
        # number of top k verb matches
        self.top_k_verb_match = 0
        # confusion matrix for perfect verb alignments
        # structure is POS - count
        self.conf_matrix = defaultdict(int) 
        
        self.k = k

    def _find_verb_nl(self, doc):
        """ 
        Finds and returns the position of the verb in the sentence
        """
        for index, token in reversed(list(enumerate(doc))):
            if token.pos_ == "VERB":
                return index

    def _find_verb_en(self, doc):
        """
        Finds and returns the position of the first verb in the sentence
        """
        for index, token in list(enumerate(doc)):
            if token.pos_ == "VERB":
                return index

    def spacy_fix(self, sentence):
        return sentence.replace("<unk>", "UNK")

    def accumulate_scores(self, source_sentence, target_sentence, attention):
        attention = attention[0]
        
        # fix so that spacy doesn't tokenize <unk> into [< , UNK, >]
        source_sentence = self.spacy_fix(source_sentence)
        target_sentence = self.spacy_fix(target_sentence)
        
        en_doc = nlp_en(source_sentence)
        en_doc = [tok for tok in en_doc]
        nl_doc = nlp_nl(target_sentence)
        nl_doc = [tok for tok in nl_doc]
        
        # find the ending verb
        verb_index_nl = self._find_verb_nl(nl_doc)
        verb_index_en = self._find_verb_en(en_doc)
        
        print("\tSentence: {}\n\tTranslation:{}".format([(tok.text, tok.pos_) for
            tok in en_doc],[(tok.text, tok.pos_) for tok in nl_doc]))

        print("\tSource Verb: {}\n\tTarget Verb:{}".format(en_doc[verb_index_en],
            nl_doc[verb_index_nl]))

        verb_attention = attention[verb_index_nl].numpy()
         
        pred_max_attention = verb_attention.argmax()

        # check for perfect match
        if pred_max_attention == verb_index_en:
            print("\tPerfect Match!")
            self.perfect_verb_match += 1
            return 

        print("\tNot a perfect match. Instead: {}".format(en_doc[pred_max_attention]))
        
        # update confusion matrix
        self.conf_matrix[en_doc[pred_max_attention].pos_] += 1
        k = self.k
        # check if it's a k/2 sized window
        start = min(0, (k-1)//2 + pred_max_attention)
        end = max(len(en_doc), (k-1)//2 + pred_max_attention)
        allowed_range = np.arange(start, end)

        if pred_max_attention in allowed_range:
            print("\tIt's in the allowed range")
            self.top_k_verb_match += 1
        print("\tIt's not in the allowed range :(")
        

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

    scorer_3 = Scorer(3)
    scorer_5 = Scorer(5)

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

            plot_attention("attentions/{}.png".format(sentence),
                           words.split(), best_pred.split(), attn[index])
            sentence += 1

            scorer_3.accumulate_scores(words, best_pred, attn[index])
            scorer_5.accumulate_scores(words, best_pred, attn[index])

    report_score('PRED', pred_score_total, pred_words_total)
