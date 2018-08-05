# De-escalation Twitter Bot
#### Aim: This project aims to explore the topics of context-free generation, topic and sentiment analysis, paraphrasing and the use of principles of non-violent communication for de-escalation of online discourse. Powered by Python machine learning backend and a Twitter bot written in Node.js (using [Seed](https://github.com/nodebox/seed "Seed GitHub Repo") for seeded context-free generation), this bot will try to participate in heated Twitter threads and resolve conflicts. 

#### Collaborators: Alexander Rossa, Frederik De Bleser

## Getting Started:
**Download**:

To download the source code and supporting files for this bot, you can just fork this repository or download it as a zip. 

The bot will still not be able to be run out of the box though, as the execution is bound to having pretrained models for topic analysis and anger analysis. Trained models and pickled tokenizers and label encoders that work with the source code in this repository can be downloaded from Amazon AWS:

- [Topic Analysis Model](https://s3-eu-west-1.amazonaws.com/seed.js/models/topic_analysis_model.h5)
- [Topic Analysis Tokenizer](https://s3-eu-west-1.amazonaws.com/seed.js/models/topic_tokenizer.pickle)
- [Topic Analysis Label Encoder](https://s3-eu-west-1.amazonaws.com/seed.js/models/topic_label_encoder.pickle)
- [Anger Analysis Model](https://s3-eu-west-1.amazonaws.com/seed.js/models/anger_analysis_model.h5)
- [Anger Analysis Tokenizer](https://s3-eu-west-1.amazonaws.com/seed.js/models/anger_tokenizer.pickle)

After downloading these, create a directory *models* in the *python-backend* subdirectory and move these models there.

### Running Locally:

This project is designed as a NodeJS frontend with Python microservices backend that takes in a tweet, analyses it using pretrained neural network models and sends back results of analysis. The backend is running on a Flask server. To run the whole project locally, firstly the Flask server needs to be launched. 

Open you command line (or terminal), navigate to the python-backed subdirectory and launch the Flask application. On Windows, this would look like this:

```
>> set FLASK_APP=app.py
>> flask run
```

On a different OS, you should be fine by just switching keyword 'set' for 'export'.

Once the Flask server is up and running, the bot can be launched and use the services provided by backend. Before we launch the bot itself, we need authentication file that the bot can use. You need to create this one yourself and call it _auth.js_. This should contain your Twitter authentication. You can get these from Twitter when creating a Twitter app. There are plenty of tutorials out there on how to do this. After you get your Twitter credentials, the _auth.js_ file should look like this:

```javascript
module.exports = {
    consumer_key: 'YOUR_CONSUMER_KEY',
    consumer_secret: 'YOUR_CONSUMER_SECRET',
    access_token: 'YOUR_ACCESS_TOKEN',
    access_token_secret: 'YOUR_ACCESS_SECRET'
}
```

With authentication successfully mastered, we can launch the bot. Open another command line window and navigate to node-frontend subdirectory. From here you can launch the bot by calling it with Node.

```t
>> node bot.js
```

That's it! The bot is now up an running. Unless you modified the source code, the bot now listens for *#DeescalationBot* hashtag and is ready to respond. 

### Remote Deployment:
Instructions for remote deployment of the bot (for example on Heroku) will be added soon. The deployment should be quite standard but needs adjusting (point to remotely stored models rather than those on disk, changing paths here and there, adjusting Procfiles and so on). If you have done this kind of thing before it should be quite straightforward and you don't have to wait for this part of tutorial to be completed.

### Customizing the Bot:

It is very likely that you'll want to customize the output of the bot or the way it responds or any of the other things there are to be customized. For this, please, look into respective subdirectories depending on what part of the bot you want to customize, as that is where the information you need is provided. If you want to adjust something with Machine Learning backend and analysis of tweets, go to the [*python-backend*](https://github.com/clips/gsoc2018/tree/master/twitter-bot/python-backend) subdirectory. If you want to change how the responses are generated or how the bot acts or what it reacts to, go to the [*node-frontend*](https://github.com/clips/gsoc2018/tree/master/twitter-bot/node-frontend) subdirectory.

## Bot Overview:

### Machine Learning Backend (Python):
The backend of the bot consists of two main scripts, residing in _python-backend_ subdirectory. These are *topic_analyser.py* and *anger_analyser.py*. These two scripts provide code for loading in a pretrained model and supporting python objects (such as Tokenizer) and they provide method *analyse(tweet)* that takes in a Tweet and returns an analysis of it. Detailed information can be found in the [*python-backend*](https://github.com/clips/gsoc2018/tree/master/twitter-bot/python-backend) subdirectory. 

### Twitter Bot Frontend (Node.js):
The frontend is fueled by Javascript. The majority of the relevant code can be seen in *bot.js* file which takes care of both communicating with backend via microservices and of finding, reading and sending out tweets. The bot relies on a couple of Node modules, most notably [seedtext](https://github.com/nodebox/seedtext), which has been co-developed along with the bot. Detailed information can be found in the [*node-frontend*](https://github.com/clips/gsoc2018/tree/master/twitter-bot/node-frontend) subdirectory.

### Seed:
Seed is a JavaScript application (also available as a Node.js module - [SeedText](https://www.npmjs.com/package/seedtext) used for seeded pseudo-random procedural content generation. It uses its own microlanguage for guiding the generation and the whole documentation for this can be seen in [its GitHub repository](https://github.com/nodebox/seed "Seed GitHub Repo"). 
