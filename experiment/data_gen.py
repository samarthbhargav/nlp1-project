import random

pos_dictionary = {
    "PRON": [
        ("i", "ik"),
        ("he", "hij"),
        ("we", "wij"),
        ("they", "zij")
    ], "AUX": [
        ("have", "heb")
    ],
    "VERB": [
        ("stolen", "gestolen")
    ],
    "SUBJECT": [
        ("the cat", "de kat"),
        ("the hond", "de hond")
    ]
}

sentences = [
    (("PRON", "AUX", "VERB", "SUBJECT"), ("PRON", "AUX", "SUBJECT", "VERB"), {
        0: 0,
        1: 1,
        2: 3,
        3: 2
    }),

]


def generate(pos_dict, sentences):
    structure_source, structure_target = random.choice(sentences).split()
    source_sentence = {}

    for index, pos in enumerate(structure_source):
        #source_sentence[index] = random.choice(pos_dict[index][])
        pass

    return " ".join(sentence) + "."


for i in range(10):
    generate(pos_dictionary, sentences)
