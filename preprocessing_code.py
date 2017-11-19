import json
import re  # 정규표현식을 처리하기 위한 모듈
import string


class FileUtils: # 유틸리티성 클래스: 데이터를 읽어들이고 데이터를 내보내는 역할
    @staticmethod  #static 이라 객체 따로 생성없이 바로 가져다 쓸 수 있음.
    def load_data(filename):
        with open(filename, 'r', encoding='UTF-8') as file:
            for line in file:
                yield json.loads(line) #yield: generator로 iteration 형태로 만들어주는 것

    @staticmethod
    def write_data(filename, output_list):
        with open(filename, 'w', encoding='UTF-8') as file:
            file.write("\n".join(output_list))


class MessageProcessor: # 문자 데이터 처리하기 위한 클래스
    def __init__(self, regex_pattern): #regex_patter: 정규표현식 패턴을 인자로 받음
        self.regex_pattern = regex_pattern

    @staticmethod
    def get_messages(posts): #대숲글 전체 데이터 중에 key 'message'를 갖는 것들만 필터링하기 위해 사용
        for post in posts:
            if 'message' in post: # post는 딕셔너리 형태이므로 찾아야하는 key값이 있는지 여부 검사
                yield post['message'] # 해당 키 값이 있는 경우 generator 생성

    def get_results(self, messages): 
        for message in messages:
            lines = message.split('\n')
            # 인덱스 3 이전에는 ~번째 외침, 시간 정보 등이 들어있기 때문에 스킵한다.
            for line in lines[3:]:
                if len(line) > 0:
                    yield self.__process_line(line)

    def __process_line(self, line):
        # 문장부호 앞뒤로 공백을 추가한다.
        s = re.sub(self.regex_pattern, r' \1 ', line) 
        # 여러개의 공백이 있는 경우엔 하나로 줄인다.
        return re.sub('\s{2,}', ' ', s)


# 실행부분
if __name__ == '__main__':
    input_filename = 'bamboo.json' #input 파일 경로 알맞게 수정
    output_filename = 'result.txt' #output 파일 경로 알맞게 수정
    punctuation_regex = '([' + string.punctuation + '])'
    bamboo_processor = MessageProcessor(punctuation_regex)

    # 데이터 읽어오기
    bamboo_data = FileUtils.load_data(input_filename)

    # 데이터 처리하기
    messages = MessageProcessor.get_messages(bamboo_data)
    results = bamboo_processor.get_results(messages)
    
    # 데이터 저장하기
    FileUtils.write_data(output_filename, results)
