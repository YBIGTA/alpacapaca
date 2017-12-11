import os
import string

from konlpy.tag import Twitter

from .preprocessing_code import FileUtils

class POSProcessor:  # 문자 데이터 처리하기 위한 클래스
    def __init__(self,
                 regex_pattern='([' + string.punctuation + '])',
                 output_dirpath='preprocess/output/',
                 tagger=None):

        self.regex_pattern = regex_pattern
        # 프로젝트 최상단 디렉토리 경로를 가져온다.
        dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.abs_output_dirpath = os.path.join(dirname, output_dirpath)
        if not os.path.exists(self.abs_output_dirpath):
            os.mkdir(self.abs_output_dirpath)
        self.tagger = tagger if tagger is not None else Twitter()

    @staticmethod
    def get_messages(posts):  # 대숲글 전체 데이터 중에 key 'message'를 갖는 것들만 필터링하기 위해 사용
        for post in posts:
            if 'message' in post:  # post는 딕셔너리 형태이므로 찾아야하는 key값이 있는지 여부 검사
                yield post['message']  # 해당 키 값이 있는 경우 generator 생성

    def get_results(self, messages):
        for message in messages:
            lines = message.split('\n')
            # 인덱스 3 이전에는 ~번째 외침, 시간 정보 등이 들어있기 때문에 스킵한다.
            for line in lines[3:]:
                if len(line) > 0:
                    yield self.process_line(line)

    def process_line(self, line):
        """한 줄의 문장을 전처리한다.

        Args:
            line: 문장 한 줄. For example: '무궁화 꽃이 피었습니다.'
        
        Returns:
            문장성분이 태깅된 문장 한 줄. For example:

            '무궁화/Noun 꽃/Noun 이/Josa 피었/Verb 습니다/Eomi ./Punctuation'
        """

        # 문장 성분을 추출한다.
        tagged_tuples = self.tagger.pos(line)
        tagged_strings = [word + '/' + tag for word, tag in tagged_tuples]

        # 문장 맨 마지막에 PAD 추가
        tagged_strings.append('<Pad>/Pad')

        # 태깅된 단어들을 이어붙인다.
        processed_line = ' '.join(tagged_strings)
        return processed_line

    def preprocess(self, input_filename, output_filename, refresh=False):
        """파일 안의 텍스트를 전처리한 후 저장한다.
        
        Args:
            input_filename: 전처리할 텍스트가 들어있는 파일의 이름
            output_filename: 전처리된 텍스트를 저장할 파일의 이름
            refresh: 저장된 파일이 있더라도 새로 전처리할 것인지 여부
        
        Returns:
            output_filepath
        """
        output_filepath = os.path.join(self.abs_output_dirpath, output_filename)

        if os.path.exists(output_filepath) and refresh is False:
            return output_filepath

        data = FileUtils.load_json_data(input_filename)
        messages = self.get_messages(data)
        results = self.get_results(messages)
        FileUtils.write_data(output_filepath, results)

        return output_filepath


if __name__ == '__main__':
    input_filename = '../collect/output/bamboo.json'
    output_filename = 'result.txt'

    preprocessor = POSProcessor()
    processed_filepath = preprocessor.preprocess(input_filename=input_filename,
                                                 output_filename=output_filename)

    print('processed filepath :', processed_filepath)
