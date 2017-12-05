from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np

class EmbeddingDataset(Dataset):
    
    def __init__(self, filepath, embedding, vector_size):
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        self.unknown = np.random.randn(vector_size)
        self.EOS = np.zeros(vector_size)
        
        self.splitted = [line.strip().split() for line in lines]
        self.cleaned = [line for line in self.splitted if len(line) > 3 and len(line) < 50] 
        self.embedding = embedding
        
    def __getitem__(self, index):
        line = self.cleaned[index]
        
        line_vector = np.stack([(self.embedding[word] if word in self.embedding else self.unknown) for word in line])
        target_list = [(self.embedding[word] if word in self.embedding else self.unknown) for word in line[1:]]
        target_vector = np.stack(target_list + [self.EOS])
        
        input_tensor = torch.FloatTensor(line_vector)
        output_tensor = torch.FloatTensor(target_vector)
        
        return input_tensor, output_tensor
    
    def __len__(self):
        return len(self.cleaned)

class EmbeddingDataLoader(DataLoader):
    
    def __init__(self, *args, **kwargs):
        super(EmbeddingDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        
    def _collate_fn(self, batch):
        max_sample = max(batch, key=lambda x: x[0].size(0))
        max_length = max_sample[0].size(0)
        embedding_size = max_sample[0].size(1)

        inputs = []
        outputs = []
        for inp, outp in batch:
            ss = inp.size(0)
            new_inp = F.pad(inp.unsqueeze(0).unsqueeze(0), (0,0,0,max_length-ss))
            new_outp = F.pad(outp.unsqueeze(0).unsqueeze(0), (0,0,0,max_length-ss))
            inputs.append(new_inp.squeeze())
            outputs.append(new_outp.squeeze())

        return torch.stack(inputs), torch.stack(outputs)