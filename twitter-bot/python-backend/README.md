# Machine Learning Backend (Python)

## Overview:

The backend consists out of multiple parts, some of which were used only for creation of the models and others that are active part of the Twitter Bot application. 

### Backend Creation:

For the Twitter bot to work correctly, it needs trained models, data for these models to be trained on and so on. All of this can be found in the subdirectories of the *python-backend* directory. There are:

- *dataset_downloader:* Contains Python source files used for downloading Tweets and making them into a dataset.
- *datasets:* Contains final datasets that were used. These are *anger_dataset.csv*, used for training of the anger analysis module and *final_dataset.csv*, used for training the topic analysis module. 
- *model_training:* In this directory, code  that was used for training and evaluating the actual models can be found. In production, both anger and topic analysis are being served by Multichannel CNN models trained by *multichannel_cnn_anger_analysis.py*  and *multichannel_cnn_topic_analysis.py*.

### Twitter Bot Backend: 

The actual backend is implemented as a microservice. For instructions on how to actually launch this, refer to the instructions in [parent directiory](https://github.com/clips/gsoc2018/tree/master/twitter-bot). Parent directory also contains an important piece of information about the need for creating *models* directory here and populating it with models, as it is what the analysers in this folder will expect. 

The microservice part is served by *app.py* which contains code for setting up Flask server and routing requests to either topic or anger analysers.

As for the analysers, both *topic_analyser.py* and *anger_analyser.py* are very similar, consisting of a definition of the model in Keras and of a method *analyse(tweet)* which takes in a tweet, encodes it into needed form, loads the respective model, makes a prediction and returns it.

## Acknowledgement:

For creation of he anger-analysing model, some data from [Hate Speech Identification](https://data.world/crowdflower/hate-speech-identification/workspace/project-summary) dataset was used. I would like to thank this way to the creators of this dataset - Thomas Davidson, Dana Warmsley, Michael Macy and Ingmar Weber - for making this dataset publicly available. The Hate Speech Identification dataset has been published as part of 

```
Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. 2017. "Automated Hate Speech Detection and the Problem of Offensive Language." Proceedings of the 11th International Conference on Web and Social Media (ICWSM). 
```

and can be found [here](https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15665).