# wsgi.py # app.py와 같은 위치
import sys
import os

CURRENT_DIR = os.getcwd()

sys.stdout = sys.stderr
sys.path.insert(0, CURRENT_DIR)

from api import app as application