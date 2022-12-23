from src import CNNReader, Corpus, Summarizer, Rouge
import sys

def analyze_summaries(target_file='res.csv', saturation_threshold=0.8, diversity_factor=6, cost_scaling_factor=0.9, words=100, randomize=False):
    print("Settings: saturation_threshold: {:f}, diversity_factor: {:f}, cost_scaling_factor: {:f}, words: {:f}, randomize: {}".format(saturation_threshold, diversity_factor, cost_scaling_factor, words, randomize))
    cnn = CNNReader('dataset/cnn/stories', randomize=True)
    print('Parsing stories...')
    docs, ref_summaries = cnn.parse_stories(limit=1000)
    print('Generating corpus...')
    corpus = Corpus(docs)

    print('Initializing ROUGE...')
    rouge = Rouge(corpus, ref_summaries)

    print('Vectorizing the corpus...')
    summarizer = Summarizer(corpus, diversity_factor=diversity_factor, random=randomize)
    print('Starting Summarization.')
    with open(target_file, 'w') as f:
        f.write("# saturation_threshold: {:f}, diversity_factor: {:f}, cost_scaling_factor: {:f}, randomized: {}\n".format(saturation_threshold, diversity_factor, cost_scaling_factor, randomize))
        for i in range(len(docs)):
            summary = summarizer.summarize_by_indices(i, budget=words, saturation_threshold=saturation_threshold, cost_scaling_factor=cost_scaling_factor)
            bigram_recall = rouge.ngram_recall(2, i, summary)
            unigram_recall = rouge.ngram_recall(1, i, summary)

            bigram_precision = rouge.ngram_precision(2, i, summary)
            unigram_precision = rouge.ngram_precision(1, i, summary)
            total_sentences = docs[i].sentences()
            summary_sentences = [total_sentences[i] for i in summary]
            summary_total_length = sum([s.raw_word_count() for s in summary_sentences])

            summary_str = [str(s) for s in summary]
            line = ','.join([docs[i].name, ' '.join(summary_str), str(summary_total_length), str(unigram_recall), str(bigram_recall), str(unigram_precision), str(bigram_precision)])
            f.write(line.strip() + "\n")
            if i % 100 == 0:
                print('processed files: ' + str(i))

if __name__ == "__main__":
    target_file = 'res.csv'
    diversity_factor = 6
    cost_scaling_factor = 0.9
    saturation_threshold = 0.8
    words = 100
    randomize = False
    if len(sys.argv) >= 2:
        target_file = sys.argv[1]
    if len(sys.argv) >= 3:
        for i in range(2, len(sys.argv), 2):
            if sys.argv[i] == '-d':
                diversity_factor = float(sys.argv[i+1])
            elif sys.argv[i] == '-c':
                cost_scaling_factor = float(sys.argv[i+1])
            elif sys.argv[i] == '-s':
                saturation_threshold = float(sys.argv[i+1])
            elif sys.argv[i] == '-w':
                words = float(sys.argv[i+1])
            elif sys.argv[i] == '-r' and sys.argv[i+1] == '1':
                randomize = True
    analyze_summaries(target_file, diversity_factor=diversity_factor, cost_scaling_factor=cost_scaling_factor, saturation_threshold=saturation_threshold, words=words, randomize = randomize)
