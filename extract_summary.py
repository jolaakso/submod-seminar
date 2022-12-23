import sys
import os
from src import CNNReader

def get_summary(filename, sentences):
    dir, file = os.path.split(filename)
    cnn_reader = CNNReader(dir)
    docs, summaries = cnn_reader.parse_by_filenames([file])
    doc, summary = (docs[0], summaries[0])
    print('Summary:')
    for i in sentences:
        print(doc.sentences()[i].original_sentence())
    print()
    print('Reference:')
    for s in summary.sentences():
        print(s.original_sentence())

if __name__ == "__main__":
    filename = sys.argv[1]
    sentences = sys.argv[2]
    get_summary(filename, [int(i) for i in sentences.split(' ')])
