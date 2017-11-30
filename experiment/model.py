import argparse
import codecs
import math
import os
import sys
from itertools import count, takewhile, zip_longest

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.abspath("../opennmt"))

from opennmt import opts
from opennmt.onmt import IO
from opennmt import onmt


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
    args = ["-model", model_path, "-src", "en_test.txt"]
    sys.argv = sys.argv[:] + args


def plot_attention(path, source_words, target_words, attention):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)

    print(attention)
    print(len(attention))

    attention = attention[0]

    print(len(source_words))
    print(len(target_words))

    cax = ax.matshow(attention.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + source_words +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + target_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(path, dpi=600)


if __name__ == '__main__':
    ### PARAMs for Model
    model = "/home/samarth/workspaces/uva/nlp1/nlp1-project/models/ted_sgd_acc_55.43_ppl_12.39_e11.pt"
    ###
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

    report_score('PRED', pred_score_total, pred_words_total)
