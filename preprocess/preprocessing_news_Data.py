import pandas as pd
import pandas as pd
import json
import re
import string
import os
from konlpy.tag import Twitter

REGEX_PATTERN = '([' + string.punctuation + '])' 
PAD_TOKEN = '<Pad>/Pad'


# 하나의 columns로 구성된 dataframe을 인자로 받아 각 row에 있는 기사 text를 한 문장 단위로 쪼갠 뒤 sereis 형태의 결과값을 리턴한다. 
def splitSentence(df):    
    resultSplit = pd.Series()
    resultSplit = pd.Series(df.map(lambda x: str(x).replace('다.', '다.^^').strip().split('^^')))
    # 기사 앞부분과 뒷부분 두 문장은 불필요하므로 제거하고 불러온다.
    resultSplit = resultSplit.map(lambda x: x[2:-2])
    resultSplit = resultSplit.map(lambda x: x if x!=[] else None).dropna(axis = 0)
    
    return resultSplit    

# 대숲 코드 차용. 각  문장에 대하여 문장부호의 앞 뒤로 공백을 추가한 결과값이 리턴된다. 
def process_line(line):
    s = re.sub(REGEX_PATTERN, r' \1 ', line) 
    return re.sub('\s{2,}', ' ', s)

# Twitter 객체를 사용하여 tagging한다. 이때 문장 끝 Pad 표시를 추가한다. 
def tagging(line):
    tagger = Twitter()
    # 문장 성분을 추출한다.
    tagged_tuples = tagger.pos(line)
    tagged_strings = [word + '/' + tag for word, tag in tagged_tuples]
    tagged_strings = [word + '/' + tag for word, tag in tagged_tuples] + [PAD_TOKEN]

    # 태깅된 단어들을 이어붙인다.
    processed_line = ' '.join(tagged_strings)
    
    return processed_line

# 처리해야하는 데이터의 구조가 이차원배열이므로 processLine() 함수를 각 문장 단위로 바로 적용할 수 없기에 실행 함수를 따로 만들었다. 
def executePL(tdLst):
    processLine = []
    for lst in tdLst:
        temp = []
        for le in lst:
            le = process_line(le)
            temp.append(le)
        processLine.append(temp)
    return processLine

# 처리해야하는 데이터의 구조가 이차원배열이므로 tagging() 함수를 각 문장 단위로 바로 적용할 수 없기에 실행 함수를 따로 만들었다. 
def executeTag(tdLst):
    tagResult = []
    for lst in tdLst:    
        temp = []         
        for line in lst:
            temp.append(tagging(line))
        tagResult.append(temp)
    return tagResult



# Foreign 이거나 Alpha, punctuation이 .," ' 이외인 것을 포함하는 문장은 제거하기 위한 함수이다. 
punctuation = ['.', "'", '"'] 
removetarget = ['Foreign', 'Alpha']
def removeTag(tdLst):
    removeResult = []    
    for lst in tdLst:
        tempSentence = []
        for eleTag in lst:
            temp = eleTag.split()
            inavailable = False
            for ele in temp:
                splitEle = ele.split('/')
                if splitEle[-1] in removetarget:
                    inavailable = True
                    break
                if splitEle[-1] == 'Punctuation' and (splitEle[0] not in punctuation):
                    inavailable = True
                    break
            if not inavailable:
                tempSentence.append(temp)
        if tempSentence != []:
            removeResult.append(tempSentence)    
    return removeResult


# 이차원배열 형태로 최종 처리 결과데이터
def write_data(filename, output_list):
        with open(filename, 'w', encoding='UTF-8') as file:
            for lst in output_list:
                for sentence in lst:
                    result = ""
                    for ele in sentence:
                        result = result + " " + ele
                        if ele.split('/')[-1] == 'Pad':
                            result = result + "\n"
                    file.write(result.strip(" ")) 
