from flask import Flask
from flask_restplus import Resource, Api
from flask_cors import CORS

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
print(parentdir)
sys.path.insert(0, parentdir) 

from generate.generator import LineGenerator
from train.trainer import DataReader
import torch

rnn = torch.load(os.path.join(parentdir, 'train/output/saved_model.pkl'))
data_reader = DataReader()
data_reader.load(os.path.join(parentdir, 'train/output/reader_params.pkl'))
generator = LineGenerator(rnn, data_reader)

app = Flask(__name__)
CORS(app)
api = Api(app)

@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world, 안녕!'}

@api.route('/alpaca/<string:input_word>')
class Alpacapaca(Resource):
    
    def get(self, input_word):
        
        results = generator.samples('가나다')

        return {'success': True,
                'results': results}

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
