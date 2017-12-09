import codecs

from data import to_onmt_format, nlp_en, nlp_nl

def process_file(file_name, target_file, nlp):
    with codecs.open(file_name, "r", "utf-8") as reader, \
            codecs.open(target_file, "w", encoding="utf-8") as writer:
        for source in reader:
            writer.write(to_onmt_format(nlp(source)))

if __name__ == "__main__":
    process_file("europalENG.txt", "europalProcessedEN.txt", nlp_en)
    process_file("europalNL.txt", "europalProcessedNL.txt", nlp_nl)
