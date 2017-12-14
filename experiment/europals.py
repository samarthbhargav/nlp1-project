import spacy

from data import is_interesting_sentence_en, is_interesting_sentence_nl, nlp_nl, nlp_en

if __name__ == '__main__':

    limit = 100

    with open("./europalENG", "r") as english, \
            open("./europalENG.txt", "w") as interestingEng, \
            open("./europalNL", 'r') as dutch, \
            open("./europalNL.txt", "w") as interestingDutch, \
            open("./europalENGUNK.txt", "w") as interestingEngUNK, \
            open("./europalNLUNK.txt", "w") as interestingNLUNK:
        count = 0



        for en_line, nl_line in zip(english, dutch):
            if not (is_interesting_sentence_en(en_line) and is_interesting_sentence_nl(nl_line)):
                continue

            interestingEng.write(en_line.strip() + "\n")
            interestingDutch.write(nl_line.strip() + "\n")

            towrite = []
            for token in list(nlp_en(en_line.strip())):
                if not token.pos_ in {"VERB", "PRON"}:
                    towrite.append("<unk>")
                else:
                    towrite.append(token.text)
            interestingEngUNK.write(' '.join(towrite) + "\n")

            towrite = []
            for token in list(nlp_nl(nl_line.strip())):
                if not token.pos_ in {"VERB", "PRON"}:
                    towrite.append("<unk>")
                else:
                    towrite.append(token.text)
            interestingNLUNK.write(' '.join(towrite) + "\n")




            count += 1
            print(count)

            if count == 5000:
                break
        print("There are {} interesting sentences".format(count))
