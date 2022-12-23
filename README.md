# Requirements

Python 3 and Pip. Required libraries are numpy, scipy and NLTK. This folder needs to have the CNN stories dataset in `dataset/cnn/stories/`. It can be downloaded here: [https://huggingface.co/datasets/cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail).

# Running

`python main.py <filename> [-d <lambda>] [-s <saturation>] [-c <cost_factor>] [-w <word count>] [-r 1]`

* lambda: The diversity factor, >= 0
* saturation: Saturation factor for coverage score, > 0, <= 1
* cost factor: Preference for shorter sentences, >= 0
* r: Select sentences completely randomly. Used for measuring a baseline for the algorith performance. When using this, cost factor should be set to 0. Set to 1 to enable.

This will generate summaries for 1000 articles in the data set. It will output a CSV file named after the argument. The CSV file columns are the file name, the indices of the selected sentences, word count of the summary, ROUGE-1 recall, ROUGE-2 recall, ROUGE-1 precision and ROUGE-2 precision.

`python parse.py <filename>`
Prints statistics of the output file of `main.py` in LaTeX table format.

`python extract_summary.py <path_to_file_in_dataset> "<summary_sentence_indices>"`

Prints out the reference summary and the generated summary of a file in data set. The indices should be separated with spaces, just like they appear in the output CSV of `main.py`.

# Notes

The project originally also included unit tests. However, they were dependent on the data set. Since I am not sure if I am authorized to host any of the data set files here, they have been omitted from this repository.
