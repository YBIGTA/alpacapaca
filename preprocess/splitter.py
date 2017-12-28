import random as random
import numpy as np
import math
import os

class RandomSplitter :
    def __init__(self, datafile, pct=0.3, output_dirpath='preprocess/train_val/') :
        
        with open(datafile, 'r', encoding='UTF-8') as file:
            data = file.readlines()
            
        self.data = data ##데이터셋 명
        self.pct = pct ## validation set 명

        # 프로젝트 최상단 디렉토리 경로를 가져온다.
        dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.abs_output_dirpath = os.path.join(dirname, output_dirpath)
        if not os.path.exists(self.abs_output_dirpath):
            os.mkdir(self.abs_output_dirpath)
        
    def split(self, train_filename="train.txt", validation_filename="validation.txt", refresh=False) :
        random.seed(0)
        random.shuffle(self.data)
        train = []
        validation = []
        standard_index = math.ceil(len(self.data)*(1-self.pct))
        
        train_filepath = os.path.join(self.abs_output_dirpath, train_filename)
        validation_filepath = os.path.join(self.abs_output_dirpath, validation_filename)

        if os.path.exists(train_filepath) and os.path.exists(validation_filepath) and not refresh:
            return train_filepath, validation_filepath

        for i in range(len(self.data)) :
            if i < standard_index  :
                train.append(self.data[i])
            else :
                validation.append(self.data[i])

        with open(train_filepath, "w", encoding = "utf-8") as file1 :
            file1.writelines(train)

        with open(validation_filepath, "w", encoding = "utf-8") as file2 :
            file2.writelines(validation)
            
        return train_filepath, validation_filepath

class TailSplitter :
    def __init__(self, datafile, pct) :
        
        with open(datafile, 'r', encoding='UTF-8') as file:
            data = file.readlines()

        self.data = data #데이터셋명
        self.pct = pct #validation set 비율
        
        # 프로젝트 최상단 디렉토리 경로를 가져온다.
        dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.abs_output_dirpath = os.path.join(dirname, output_dirpath)
        if not os.path.exists(self.abs_output_dirpath):
            os.mkdir(self.abs_output_dirpath)
        
    def split(self, train_filename="train.txt", validation_filename="validation.txt"):
        train = []
        validation = []
        standard_index = math.ceil(len(self.data)*(1-self.pct))

        train_filepath = os.path.join(self.abs_output_dirpath, train_filename)
        validation_filepath = os.path.join(self.abs_output_dirpath, validation_filename)

        if os.path.exists(train_filepath) and os.path.exists(validation_filepath) and not refresh:
            return train_filepath, validation_filepath

        for i in range(len(self.data)) :
            if i < standard_index :
                train.append(self.data[i])
            else :
                validation.append(self.data[i])

        with open(train_filepath, "w", encoding = "utf-8") as file1 :
            file1.writelines(train)

        with open(validation_filepath, "w", encoding = "utf-8") as file2 :
            file2.writelines(validation)
            
        return train_filepath, validation_filepath
