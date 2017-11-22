import codecs

import re
import spacy

import time
from experiment import text_utils

nlp = spacy.load("nl")


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


def is_interesting_sentence(sentence):
    doc = nlp(sentence)
    n = len(doc)
    if n <= 5 or n > 40:
        return False
    for token in list(doc)[::-1]:
        if token.is_alpha:
            return token.pos_ == "VERB"


if __name__ == '__main__':

    count = 0
    verb_count = 0
    start_time = time.time()
    with open("test.txt", "w") as writer:
        for index, (s, t) in enumerate(TedTalksDataSet("../DeEnItNlRo-DeEnItNlRo/train.tags.en-nl.en",
                                                       "../DeEnItNlRo-DeEnItNlRo/train.tags.en-nl.nl")):

            if is_interesting_sentence(t):
                writer.write(t + "\n\n")
                verb_count += 1
            count += 1

            if verb_count > 100000:
                break

            if index % 1000 == 0:
                print(index + 1, " done. Minutes elapsed: ", (time.time() - start_time) / 60, "\t", verb_count,
                      "found so far!")

    print(count)
    print(verb_count)
    ...
