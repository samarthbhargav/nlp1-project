import spacy

from data import is_interesting_sentence_en, is_interesting_sentence_nl, nlp_nl, nlp_en

if __name__ == '__main__':

    limit = 100

    with open("../data/nl-en/europarl-v7.nl-en.en", "r") as english, \
            open("./europalENG.txt", "w") as interestingEng, \
            open("../data/nl-en/europarl-v7.nl-en.nl", 'r') as dutch, \
            open("./europalNL.txt", "w") as interestingDutch:
        count = 0

        for en_line, nl_line in zip(english, dutch):
            if not (is_interesting_sentence_en(en_line) and is_interesting_sentence_nl(nl_line)):
                continue

            interestingEng.write(en_line.strip() + "\n")
            interestingDutch.write(nl_line.strip() + "\n")

            count += 1
            print(count)

        print("There are {} interesting sentences".format(count))
