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
    def __init__(self, num_features=500, min_word_count=10, num_workers=4, context=10, downsampling=1e-5, sg=1):
        self.num_features = num_features
        self.min_word_count = min_word_count
        self.num_workers = num_workers
        self.context = context # window size
        self.downsampling = downsampling
        # skip gram이 좋은 이유 : https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/30/word2vec/
        self.sg = sg
        
    def train_model(self, words):
        self.model = Word2Vec(words, 
                         num_features=self.num_features,
                         min_word_count=self.min_word_count, 
                         num_workers=self.num_workers,
                         context=self.context, 
                         downsample=self.downsampling, 
                         sg=self.sg)
  
        # 더 train 시킬 생각 없을 때. 필요 없는 메모리를 unload 시킨다.
        self.model.init_sims(replace=True)

    def save(self, model_filename):
        self.model.save(model_filename)
    
    def load(self, model_filename):
        self.model = Word2Vec.load(model_filename)

    def build_embedding(self, input_filename, model_filename):

        self.words = TextUtils.read_words(input_filename)
        self.train_model(self.words)
        self.save(model_filename)

if __name__ == '__main__':
    
    input_filename = 'result.txt'
    model_filename = 'word2vec_model'
    model = Word2VecModel()    
    model.build_embedding(input_filename=input_filename, model_filename=model_filename)