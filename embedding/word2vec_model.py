from gensim.models import Word2Vec
import pickle as p

class Word2VecModel() :
    def __init__(self, num_features, min_word_count, num_workers, context, downsampling, sg) :
        self.num_features = num_features
        self.min_word_count = min_word_count
        self.num_workers = num_workers
        self.context = context
        self.downsampling = downsampling
        self.sg = sg
        
        
    def read_data(self, filename) :
        with open(filename, 'r', encoding= "utf-8") as file :
            sentences = file.readlines()
        words = []
        for sent in sentences :
            words.append(sent.strip().split())
        return words
    
    
    def build_embedding(self, filename) :
        words = self.read_data(filename)
        model = Word2Vec(words, 
                         num_features= self.num_features,
                         min_word_count= self.min_word_count, 
                         num_workers= self.num_workers,
                         context= self.context, 
                         downsample= self.downsampling, 
                         sg = self.sg)
 
        model.init_sims(replace=True)
        self.model = model
        
    def save(self, filename):
        self.model.save(filename)
    
    
    def load(self, filename):
        self.model = Word2Vec.load(filename)
