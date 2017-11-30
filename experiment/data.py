import codecs
import time

import spacy
from lxml import etree

nlp_nl = spacy.load("nl")
nlp_en = spacy.load("en")


class TedTalksDataSet:
    def __init__(self, source_file, target_file, encoding="utf-8"):
        self.source_file = source_file
        self.target_file = target_file
        self.encoding = encoding

    def _parse_single_doc(self, doc):
        lines = []
        for idx, line in enumerate(doc.split("\n")):
            if "<" in line:
                continue
            lines.append(line)
        return lines

    def __iter__(self):
        with codecs.open(self.source_file, encoding=self.encoding) as source_file, \
                codecs.open(self.target_file, encoding=self.encoding) as target_file:
            source = ""
            target = ""
            for source_line, target_line in zip(source_file, target_file):
                source += source_line
                target += target_line

                if "</doc>" in source_line:
                    ss = self._parse_single_doc(source.strip())
                    tt = self._parse_single_doc(target.strip())
                    assert len(ss) == len(tt)
                    for s, t in zip(ss, tt):
                        yield s, t


def to_onmt_format(doc):
    tokens = []
    for token in doc:
        token = token.text
        if "'" in token:
            token = token.replace("'", "&apos;")
        tokens.append(token)
    return " ".join(tokens)


def is_interesting_sentence_en(sentence):
    doc = nlp_en(sentence.strip())
    starts = False
    next_pos = None
    n = len(doc)
    if n <= 5 or n > 40:
        return False

    doc = list(doc)

    # print([d.pos_ for d in doc[:3]])

    if doc[0].pos_ == "DET" and doc[1].pos_ == "NOUN":
        starts = True
        next_pos = 2
    if doc[0].pos_ in {"NOUN", "PRON"}:
        starts = True
        next_pos = 1

    if not starts:
        return False

    if doc[next_pos].pos_ in {"AUX", "VERB"} and doc[next_pos + 1].pos_ == "VERB":
        return True


def is_interesting_sentence_nl(sentence):
    doc = nlp_nl(sentence)
    n = len(doc)
    if n <= 5 or n > 40:
        return False
    doc = list(doc)
    for token in doc[::-1]:
        if token.is_alpha:
            return token.pos_ == "VERB"


def get_parallel_test(source_lang_file, target_lang_file):
    with codecs.open(source_lang_file, "r", encoding="utf-8") as source_reader, \
            codecs.open(target_lang_file, "r", encoding="utf-8") as target_reader:
        source_lang = etree.XML(bytes(bytearray(source_reader.read(), encoding='utf-8')))
        target_lang = etree.XML(bytes(bytearray(target_reader.read(), encoding='utf-8')))

    count = 0
    for source_doc, target_doc in zip(source_lang[0], target_lang[0]):
        for sourge_seg, target_seg in zip(source_doc, target_doc):
            if sourge_seg.tag == "seg":
                count += 1
                yield (sourge_seg.text.strip(), target_seg.text.strip())

    print("Total number of sentences: {}".format(count))


def read_train_file():
    count = 0
    verb_count = 0
    start_time = time.time()

    with open("test.txt", "w") as writer:
        for index, (s, t) in enumerate(TedTalksDataSet("../DeEnItNlRo-DeEnItNlRo/train.tags.en-nl.en",
                                                       "../DeEnItNlRo-DeEnItNlRo/train.tags.en-nl.nl")):

            if is_interesting_sentence_en(s):
                writer.write(s + "\n" + t + "\n\n")
                verb_count += 1

            count += 1

            if verb_count > 5000:
                break

            if index % 1000 == 0:
                print(index + 1, " done. Minutes elapsed: ", (time.time() - start_time) / 60, "\t", verb_count,
                      "found so far!")

    print(count)
    print(verb_count)


def filter_interesting(tup):
    source, target = tup
    return is_interesting_sentence_en(source)


if __name__ == '__main__':
    with open("en_test.txt", "w") as en_writer, open("nl_test.txt", "w") as nl_writer:

        gen = get_parallel_test("../DeEnItNlRo-DeEnItNlRo/IWSLT17.TED.tst2010.en-nl.en.xml",
                                "../DeEnItNlRo-DeEnItNlRo/IWSLT17.TED.tst2010.en-nl.nl.xml")
        filtered = filter(filter_interesting, gen)
        count = 0
        for s, t in filtered:
            count += 1
            en_writer.write(to_onmt_format(nlp_en(s)) + "\n")
            nl_writer.write(to_onmt_format(nlp_nl(t)) + "\n")

        gen = get_parallel_test("../DeEnItNlRo-DeEnItNlRo/IWSLT17.TED.dev2010.en-nl.en.xml",
                                "../DeEnItNlRo-DeEnItNlRo/IWSLT17.TED.dev2010.en-nl.nl.xml")

        filtered = filter(filter_interesting, gen)
        for s, t in filtered:
            count += 1
            en_writer.write(to_onmt_format(nlp_en(s)) + "\n")
            nl_writer.write(to_onmt_format(nlp_nl(t)) + "\n")

        print(count)
