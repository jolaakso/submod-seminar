# Requirements

Python 3 and Pip. Required libraries are numpy, scipy and NLTK. This folder needs to have the CNN stories dataset in `dataset/cnn/stories/`. It can be downloaded here: [https://huggingface.co/datasets/cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail).

# Running

`python main.py <filename> [-d <lambda>] [-s <saturation>] [-c <cost_factor>] [-w <word count>] [-r 1]`

* lambda: The diversity factor, >= 0
* saturation: Saturation factor for coverage score, > 0, <= 1
* cost factor: Preference for shorter sentences, >= 0
* r: Select sentences completely randomly. Used for measuring a baseline for the algorith performance. When using this, cost factor should be set to 0. Set to 1 to enable.

# Notes

The project originally also included unit tests. However, they were dependent on the data set. Since I am not sure if I am authorized to host any of the data set files here, they have been omitted from this repository.
