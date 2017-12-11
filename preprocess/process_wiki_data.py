import logging

from os import listdir
from os.path import isfile, join
from konlpy.tag import Twitter
from preprocess.bamboo_processor import POSProcessor
from preprocess.preprocessing_code import FileUtils

logger = logging.getLogger()

if __name__ == '__main__':
    # firstly, wiki dump data is processed by WikiExtractor (https://github.com/attardi/wikiextractor)
    # `input_directory` contains output of WikiExtractor
    input_directory = 'CHANGE_IT_TO_INPUT_DIRECTORY_PATH'
    tagging_module = Twitter()
    POSProcessor = POSProcessor(tagger=tagging_module)
    wiki_data_files = [f for f in listdir(input_directory) if isfile(join(input_directory, f))]

    for filename in wiki_data_files:
        wiki_data = FileUtils.load_wiki_data(join(input_directory, filename))
        processed_data = [POSProcessor.process_line(line) for line in wiki_data]
        filtered = [line for line in processed_data if 'Foreign' not in line]
        FileUtils.write_data(filename + ".txt", filtered)
