
# coding: utf-8

# In[1]:

from gensim.models import word2vec
import pickle as p


# In[2]:

f = open('result.txt', encoding = 'utf-8')
sentences = f.readlines()


# In[3]:

word = []
for sent in sentences:
    word.append(sent.strip().split())


# # parameter 설명
# ---
# 1. num_features = Word vector dimensionality
# 2. num_word_count = Minimun word count
# 3. num_workers 
#     - Cython이 설치됬을 때에만 지정하는 게 의미 있음. 설치되지 않았으면 무조건 1core임. 데이터 트레이닝의 속도를 높이기 위한 병렬 처리 시 core 갯수
#     - Number of threads to run in parallel
# 4. context  : Context window size 
# 5. downsampling : Downsample setting for frequent words
# 6. size : output의 단어가 표현될 vector의 dimensionality
#  

# In[5]:

num_features = 500 # 500 차원.
min_word_count = 10 # 일단 10으로 설정. 원래는 40이었는데 단어가 너무 줄어들 것 같아서 그냥 넣었다. 논리 없음.
num_workers = 4  #뭔지 모르지만 4로 설정되어 있었다. 한 번에 돌아가는 thread 갯수..?
context = 10  #window size 일단 10으로 설정!
downsampling = 1e-3  # downsampling ㄹㅇ 모름

print("Training model...")

#sg = 1 추가해서 skip-gram을 사용했다.
#왜 skip-gram이 더 좋은 지는 여기 https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/30/word2vec/  에 나와 있길래!

#변수 설정을 정말 뇌피셜로 했다. 용래님이 알아서 수정해주실 거라고 믿어 의심치 않는다.
model = word2vec.Word2Vec(word, workers=num_workers,size=num_features, min_count=min_word_count,window=context, sample=downsampling, sg = 1)


# 더 train 시킬 생각 없을 때만 써야한당. 필요 없는 메모리를 unload 시키는 메소드.
model.init_sims(replace=True) 

model.save("Word2Vec_test")

#RAM 8GB SSD 256GB intel i5인 내 컴에서도 7-8분 내외로 돌아간다.


# In[6]:

model.most_similar(positive = ['안녕'], topn = 30)


# In[ ]:



