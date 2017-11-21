import codecs

import re

from experiment import text_utils


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


verb_pattern = re.compile(r"(en|d|t)$")


def is_ending_with_verb(sentence):
    tokens = text_utils.split_sentence(text_utils.normalize(sentence))
    if len(tokens) > 1 and verb_pattern.search(tokens[-1]) is not None:
        return True
    if len(tokens) > 2 and verb_pattern.search(tokens[-2]) is not None:
        return True
    return False


if __name__ == '__main__':
    count = 0
    verb_count = 0
    with open("test.txt", "w") as writer:
        for s, t in TedTalksDataSet("../DeEnItNlRo-DeEnItNlRo/train.tags.en-nl.en",
                                    "../DeEnItNlRo-DeEnItNlRo/train.tags.en-nl.nl"):
            writer.write("{}\n{}\n\n".format(s.strip(), t.strip()))
            if is_ending_with_verb(t):
                # print(t)
                verb_count += 1
            count += 1
    print(count)
    print(verb_count)
