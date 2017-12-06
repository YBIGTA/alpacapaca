from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np


class EmbeddingDataset(Dataset):
    def __init__(self, filepath, embedding, vector_size, sampleSize=None):

        with open(filepath, 'r') as f:
            if sampleSize is not None:
                lines = f.readlines()[1:sampleSize]
            else:
                lines = f.readlines()

        self.unknown = np.random.randn(vector_size)
        self.EOS = np.zeros(vector_size)

        self.splitted = [line.strip().split() for line in lines]
        self.cleaned = [line for line in self.splitted if len(line) > 3 and len(line) < 50]
        self.cleaned = self.cleaned[: int(len(self.cleaned) / 10) * 10]
        self.embedding = embedding
        print('total read line %d' % len(lines))
        print('cleaned lines %d' % len(self.cleaned))

    def __getitem__(self, index):
        line = self.cleaned[index]

        line2EmbeddingLine = [(self.embedding[word] if word in self.embedding else self.unknown) for word in line]
        line_vector = np.stack(line2EmbeddingLine)
        target_list = line2EmbeddingLine[1:]
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
            new_inp = F.pad(inp.unsqueeze(0).unsqueeze(0), (0, 0, 0, max_length - ss))
            new_outp = F.pad(outp.unsqueeze(0).unsqueeze(0), (0, 0, 0, max_length - ss))
            inputs.append(new_inp.squeeze())
            outputs.append(new_outp.squeeze())

        if len(inputs) != self.batch_size:
            for _ in range(0, self.batch_size - len(inputs)):
                inputs.append(torch.zeros(max_length, embedding_size))

        if len(outputs) != self.batch_size:
            for _ in range(0, self.batch_size - len(outputs)):
                outputs.append(torch.zeros(max_length, embedding_size))

        return torch.stack(inputs), torch.stack(outputs)
