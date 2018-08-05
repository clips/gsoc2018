#!/bin/bash
git clone https://github.com/clips/gsoc2018.git
cd gsoc2018/gdpr/
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -c "import nltk; nltk.download('wordnet')"
cd privacy
echo "downloading wordvectors, will take time"
wget http://magnitude.plasticity.ai/glove+approx/glove.6B.100d.magnitude
SKIP_MAGNITUDE_WHEEL=1 pip3 install pymagnitude==0.1.46
python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py runserver