import re

norm_patterns = [
    (re.compile(r"\(\w+\)"), " "),  # remove words in braces like (Music) etc
    (re.compile(r"([.!?])"), r" \1"),  # replace punctuation with a punctuation + a space before it
    (re.compile(r"[^a-zA-Z.!?]+"), r" "),  # replace non alphabet / punctuation with spaces
]


def normalize(s):
    for pattern, replacement in norm_patterns:
        s = pattern.sub(replacement, s)
    return s


def split_sentence(s):
    return s.split()


if __name__ == '__main__':
    print(split_sentence(normalize("The name of the person is Pascal (Applause).")))

    print()
