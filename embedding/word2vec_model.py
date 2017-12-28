from gensim.models import Word2Vec
import numpy as np
import os

class TextUtils:
    @staticmethod
    def read_words(filename):
        with open(filename, 'r', encoding= "utf-8") as file:
            return [sentence.strip().split() for sentence in file.readlines()]

class Word2VecModel() :
    def __init__(self, size=300, window=1, min_count=10, workers=4, sg=1, sample=1e-5, iter=20,
                 output_dirpath='embedding/output/'):
        self.size = size
        self.window = window # window size
        self.min_count = min_count
        self.workers = workers
        self.sample = sample # downsampling
        self.iter = iter
        # skip gram이 좋은 이유 : https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/30/word2vec/
        self.sg = sg

        self.unknown_token = np.random.randn(self.size)
        self.vocab = None

        # 프로젝트 최상단 디렉토리 경로를 가져온다.
        dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.abs_output_dirpath = os.path.join(dirname, output_dirpath)
        if not os.path.exists(self.abs_output_dirpath):
            os.mkdir(self.abs_output_dirpath)

    def build_embedding(self, input_filename, output_filename, refresh=False):

        output_filepath = os.path.join(self.abs_output_dirpath, output_filename)
        
        if (os.path.exists(output_filepath + '.w2v') 
            and os.path.exists(output_filepath + '_unknown_token.npy')
            and not refresh):

            return self.load(output_filepath)

        words = TextUtils.read_words(input_filename)
        self.model = Word2Vec(words,
                         size=self.size,
                         window=self.window, 
                         min_count=self.min_count,
                         workers=self.workers, 
                         sample=self.sample, 
                         sg=self.sg,
                         iter = self.iter)
        self.vocab = self.model.wv.vocab.keys()
        self.pad_token = self.model.wv['<Pad>/Pad']

        self.save(output_filepath)

    def vectorize(self, word):
        return self.model.wv[word] if word in self.model.wv else self.unknown_token

    def save(self, output_filepath):
        self.model.save(output_filepath + '.w2v')
        np.save(output_filepath + '_unknown_token.npy', self.unknown_token)
    
    def load(self, output_filepath):
        self.model = Word2Vec.load(output_filepath + '.w2v')
        self.unknown_token = np.load(output_filepath + '_unknown_token.npy')
        self.vocab = self.model.wv.vocab.keys()
        self.pad_token = self.model.wv['<Pad>/Pad']
    
    @classmethod
    def load_from(cls, output_filename, output_dirpath='embedding/output/'):
        
        # 프로젝트 최상단 디렉토리 경로를 가져온다.
        dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_output_dirpath = os.path.join(dirname, output_dirpath)
        output_filepath = os.path.join(abs_output_dirpath, output_filename)
        
        instance = cls()
        instance.model = Word2Vec.load(output_filepath + '.w2v')
        instance.unknown_token = np.load(output_filepath + '_unknown_token.npy')
        instance.vocab = instance.model.wv.vocab.keys()
        instance.pad_token = instance.model.wv['<Pad>/Pad']
        return instance
        
class MultiWord2VecModel():
    def __init__(self, size=300, window=1, min_count=10, workers=4, sg=1, sample=1e-5, iter=20,
                 output_dirpath='embedding/output/'):
        self.size = size
        self.window = window # window size
        self.min_count = min_count
        self.workers = workers
        self.sample = sample # downsampling
        self.iter = iter
        # skip gram이 좋은 이유 : https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/30/word2vec/
        self.sg = sg

        self.unknown_token = np.random.randn(self.size)
        self.vocab = None

        # 프로젝트 최상단 디렉토리 경로를 가져온다.
        dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.abs_output_dirpath = os.path.join(dirname, output_dirpath)
        if not os.path.exists(self.abs_output_dirpath):
            os.mkdir(self.abs_output_dirpath)

    def build_embedding(self, input_filenames, output_filename, refresh=False):

        output_filepath = os.path.join(self.abs_output_dirpath, output_filename)
        
        if (os.path.exists(output_filepath + '.w2v') 
            and os.path.exists(output_filepath + '_unknown_token.npy')
            and not refresh):

            return self.load(output_filepath)
        
        all_words = []
        for input_filename in input_filenames:
            words = TextUtils.read_words(input_filename)
            all_words += words
            
        self.model = Word2Vec(all_words,
                         size=self.size,
                         window=self.window, 
                         min_count=self.min_count,
                         workers=self.workers, 
                         sample=self.sample, 
                         sg=self.sg,
                         iter = self.iter)
        self.vocab = self.model.wv.vocab.keys()
        self.pad_token = self.model.wv['<Pad>/Pad']

        self.save(output_filepath)

    def vectorize(self, word):
        return self.model.wv[word] if word in self.model.wv else self.unknown_token

    def save(self, output_filepath):
        self.model.save(output_filepath + '.w2v')
        np.save(output_filepath + '_unknown_token.npy', self.unknown_token)
    
    def load(self, output_filepath):
        self.model = Word2Vec.load(output_filepath + '.w2v')
        self.unknown_token = np.load(output_filepath + '_unknown_token.npy')
        self.vocab = self.model.wv.vocab.keys()
        self.pad_token = self.model.wv['<Pad>/Pad']
    
    @classmethod
    def load_from(cls, output_filename, output_dirpath='embedding/output/'):
        
        # 프로젝트 최상단 디렉토리 경로를 가져온다.
        dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_output_dirpath = os.path.join(dirname, output_dirpath)
        output_filepath = os.path.join(abs_output_dirpath, output_filename)
        
        instance = cls()
        instance.model = Word2Vec.load(output_filepath + '.w2v')
        instance.unknown_token = np.load(output_filepath + '_unknown_token.npy')
        instance.vocab = instance.model.wv.vocab.keys()
        instance.pad_token = instance.model.wv['<Pad>/Pad']
        return instance
    
if __name__ == '__main__':
    
    input_filename = '../preprocess/output/result.txt'
    model_filename = 'word2vec_model'
    model = Word2VecModel()    
    model.build_embedding(input_filename=input_filename, output_filename=model_filename)