from preprocess.bamboo_processor import POSProcessor
from embedding.word2vec_model import Word2VecModel
from data_load.embedding_loader import EmbeddingDataset, EmbeddingDataLoader
from model.rnn import BasicLSTM
from train.base_trainer import DataLoaderTrainer
from generate.base_generator import LineEmbeddingGenerator

import os
from torch.optim import Adam, lr_scheduler

if __name__ == '__main__':

	raw_datafile = 'collect/output/bamboo.json'
	preprocessor = POSProcessor()
	processed_filepath = preprocessor.preprocess(input_filename=raw_datafile, 
	                                             output_filename='result.txt',
	                                             refresh=False)
	print('Done preprocessing')

	embedding = Word2VecModel()
	embedding.build_embedding(input_filename=processed_filepath, 
							  output_filename='bamboo_word2vec', 
                          	  refresh=False)
	print('Done embedding')
	
	dataset = EmbeddingDataset(input_filepath=processed_filepath, embedding=embedding)
	dataloader = EmbeddingDataLoader(dataset, batch_size=64)
	
	model = BasicLSTM(input_size=100, rnn_size=128, hidden_size=200, target_size=100)


	optimizer = Adam(model.parameters())
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	trainer = DataLoaderTrainer(model, dataloader, optimizer, exp_lr_scheduler)

	trainer.run(epochs=1)

	print('Done training')
	generator = LineEmbeddingGenerator(model=model, embedding=embedding)

	print('"알파카"로 삼행시')
	poem = generator.samples('알파카')
	print('\n'.join(poem))
	print()

	print('"아이유"로 삼행시')
	poem = generator.samples('아이유')
	print('\n'.join(poem))
	print()

	print('"김연아"로 삼행시')
	poem = generator.samples('김연아')
	print('\n'.join(poem))
	print()