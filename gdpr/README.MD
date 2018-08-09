# GDPR Anonymization Tool
This tool is built to aid companies/organizations in their text anonymization needs.
##### What is text anonymization?
Text Anonymization refers to the processing of text, stripping it of any attributes/identifiers thus hiding sensitive details and protecting the identity of users.
## Usage Guide:  [Click Here](https://github.com/clips/gsoc2018/tree/master/gdpr/USAGE_GUIDE)

## Architecture
This system consists of two main components.

 ### Sensitive Attribute Detection System
Before text can be anonymized, the sensitive attributes in the text which give out information have to be identified. We use two methods to do the same (that can be used in tandem or as standalone systems as well)

 1. **Named Entity Recognition Based Detection**: This relies on tagging of tagging of sensitive entities in text. The user can setup different configurations for different entities which would determine how the given entity anonymized. The different options available are:  Deletion/Replacement, Supression and Generalization. The system currently ships with [Spacy's](https://spacy.io/) NER system, but can very easily be switched out for other NER models. 
 2. **TF-IDF Based rare entity detection**: Certain sensitive attributes in text might not neccesarily be tagged/identified by the NER system. These sensitive tokens can be identified by the TF-IDF system. The  term frequency–inverse document frequency identifies possible rare entities in text based on the distribution and occurence of tokens across sample text snippets supplied by the user. Once the TF-IDF score threshold is set, tokens with scores above the same are determined to be sensitive and anonymized.

 ### Sensitive Attribute Anonymization System
 Once the sensitive attributes/tokens are detected, they need to be anonymized depening on the kind of token they are. The user can set different anonymization actions for different tokens. The currently available options are:
 1.  **Deletion/Replacement**: To be used in certain cases where retaining a part of the attribute through the other anonymization methods too is not appropriate. Completely replaces the attribute with a pre-set replacement. 
 Example: `My name is John Doe` would be replaced by `My name is <Name>`.
 2. **Supression**:  To be used when hiding a part of the information is enough to protect the user's anonymity. The user can supply the percentage or the number of bits they want to be supressed. 
 Example: `My phone number is 9876543210` would be replaced by `My phone number is 98675*****` if the user chooses 50% supression. 
 3. **Generalization**: To be used when the entity is sensitive enough to need anonymization but can still be partially retained to provide information. This system has two methods of carrying out generalization
    * Word Vector Based: In this option of generalization, the nearest neighbor of the word in the vector space is utlilized to generalize the attribute. 
Example: `I live in India` get's generalized to `I live in Pakistan`. 
This method, while completely changing the word largely retains vector space information useful in most NLP and Text Processing tasks
    * Part Holonym Based: In this option, the system parses the [Wordnet Lexical Database](https://wordnet.princeton.edu/) to extract part holonyms. This method works exceptionally well with geographical entities. In this, the user can choose the level of generalization. 
Example: `I live in Beijing` get's generalized to `I live in China` at level 1 generalization and to `I live in Asia` at level 2 of generalization.

## Installation / Setup
#### By using a Setup Script
You may directly use a setup script to complete all the steps. Currently tested with a Ubuntu 16.04, Python 3.6 setup. Not recommended because of the very large downloads which may fail. Find the setup script [here](https://github.com/clips/gsoc2018/blob/master/gdpr/setup.sh).  
#### By following instructions (recommended) 
 1. Pull the code from github using `git clone https://github.com/clips/gsoc2018.git`
 2. Traverse into the GDPR directory using `cd gsoc2018/gdpr/`
 3. Install the requirments using `pip3 install -r requirements.txt`. 
 4. You also need to download the spacy model. Download and install it using `python3 -m spacy download en_core_web_sm`. Should not take more a few minutes (29 MB of download).
 5. Download NLTK corpus by using `python3 -c "import nltk; nltk.download('wordnet')"`
 6.  Download the [Plasticity AI's](https://github.com/plasticityai/magnitude) "magnitude" format of wordvectors by clicking [here](http://magnitude.plasticity.ai/glove+approx/glove.6B.100d.magnitude). (To be used in Wordvector based generalization)
 7. Paste the file in the privacy directory.  (You may also do this in one step by doing `cd privacy` followed by 
`wget http://magnitude.plasticity.ai/glove+approx/glove.6B.100d.magnitude` 
 8. Install the magnitude package by doing: `SKIP_MAGNITUDE_WHEEL=1 pip3 install pymagnitude==0.1.46` (not there in the requirements because of the extra flag)
 9. In case you haven't already, go into the privacy directory by doing `cd privacy` and run the following commands : `python3 manage.py makemigrations` and `python3 manage.py migrate`. 
 10. Your application is now ready to use. Run `python3 manage.py runserver` to start it up and navigate in your console to [localhost:8000](localhost:8000) to view the webapp.
11. If you are deploying the app on a server that needs to be reached externally, add your hostnames in `ALLOWED_HOSTS =` in the filee privacy/settings.py and run `python3 manage.py runserver 0.0.0.0:8000` (or whatever port you want to run it on).
