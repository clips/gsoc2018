# GDPR Anonymization Tool
## Instructions to setup

 1. Pull the code from github using `git clone https://github.com/clips/gsoc2018.git`
 2. Traverse into the GDPR directory using `cd gsoc2018/gdpr/`
 3. Install the requirments using `pip3 install -r requirements.txt`. 
 4. You also need to download the spacy model. Download and install it using `python3 -m spacy download en_core_web_sm`. Should not take more a few minutes (29 MB of download).
 5. Go into the privacy directory by doing `cd privacy` and run the following commands : `python3 manage.py makemigrations` and `python3 manage.py migrate`. 
 6. Your application is now ready to use. Run `python3 manage.py runserver` to start it up and navigate in your console to [localhost:8000](localhost:8000) to view the webapp.