# config.py
SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:root@localhost/dhrp_db'
SQLALCHEMY_TRACK_MODIFICATIONS = False

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")