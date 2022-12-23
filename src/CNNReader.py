import re
import glob
import os
import random
from . import Document
from . import Sentence

class CNNReader:
    parser_re = re.compile(r"(.+)(?:@highlight (.+))*", re.MULTILINE)
    def __init__(self, stories_dir, randomize=False, random_seed=0xBEEFBEEF):
        self.randomize = randomize
        self.random_seed = random_seed
        self.stories_dir = stories_dir

    def parse_stories(self, limit=-1):
        docs = []
        summaries = []
        file_names = glob.glob(os.path.join(self.stories_dir, '*.story'))
        if self.randomize:
            random.Random(self.random_seed).shuffle(file_names)
        if limit > -1:
            file_names = file_names[:limit]
        for story_file_name in file_names:
            with open(os.path.join(os.getcwd(), story_file_name), 'r', encoding="utf-8") as story_file:
                doc, summary = self.parse_story(story_file)
                if doc and summary:
                    docs.append(doc)
                    summaries.append(summary)
        return (docs, summaries)

    def parse_by_filenames(self, file_names):
        docs = []
        summaries = []
        for story_file_name in file_names:
            with open(os.path.join(self.stories_dir, story_file_name), 'r', encoding="utf-8") as story_file:
                doc, summary = self.parse_story(story_file)
                if doc and summary:
                    docs.append(doc)
                    summaries.append(summary)
        return (docs, summaries)

    # Output: document of the story, list of sentences that are the summary
    def parse_story(self, file):
        in_highlights = False
        story_lines = []
        summary_lines = []
        line_num = 0
        for line in file.readlines():
            if line_num == 0:
                line = re.sub(r"^.*\(CNN\) -- ", '', line)
                line = re.sub(r"^\s*\(CNN\)\s*", '', line)
            if re.match(r"\W*$", line):
                pass
            elif re.match(r"@highlight", line):
                in_highlights = True
            elif in_highlights:
                summary_lines.append(line.strip())
            else:
                story_lines.append(line.strip())
            line_num += 1
        if len(story_lines) == 0 or len(summary_lines) == 0:
            return (None, None)
        story_text = ' '.join(story_lines)
        basename = os.path.basename(file.name)


        return (Document(story_text, name=basename), Document(summary_lines, name='summary-' + file.name))
