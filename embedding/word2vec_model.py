from gensim.models import Word2Vec

class TextUtils:
    @staticmethod
    def read_words(filename):
        with open(filename, 'r', encoding= "utf-8") as file:
            sentences = file.readlines()
        words = []
        for sent in sentences :
            words.append(sent.strip().split())
        return words

class Word2VecModel() :
    def __init__(self, size=100, window=2, min_count=10, workers=4, sg=1, sample=1e-5):
        self.size = size
        self.window = window # window size
        self.min_count = min_count
        self.workers = workers
        self.sample = sample
        # skip gram이 좋은 이유 : https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/30/word2vec/
        self.sg = sg
        
    def build_embedding(self, input_filename):
        words = TextUtils.read_words(input_filename)
        self.model = Word2Vec(words, 
                         size=self.size,
                         window=self.window, 
                         min_count=self.min_count,
                         workers=self.workers, 
                         sample=self.sample, 
                         sg=self.sg)

    def save(self, model_filename):
        self.model.save(model_filename)
    
    def load(self, model_filename):
        self.model = Word2Vec.load(model_filename)

if __name__ == '__main__':
    
    input_filename = '../preprocess/output.result.txt'
    model_filename = 'output/word2vec_model'
    model = Word2VecModel()    
    model.build_embedding(input_filename=input_filename)
    model.save(model_filename=model_filename)