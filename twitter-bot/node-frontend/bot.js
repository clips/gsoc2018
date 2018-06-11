// const Seed = require('seed-emrg');
const spawn = require('child_process').spawn;
const Twit = require('twit')
//Getting twitter configuration from auth.js file
const auth = require('./auth.js')
// Twitter Bot object to connect to Twitter API
const T = new Twit(auth);

var stream = T.stream('statuses/filter', { track: ['#DeescalationBotCallToAction, #DeeBCTA42']});
var tweetToAnalyse = 'Sample tweet to be gotten by the bot';


// Listen for a call to action and respond in thread to it.
var responseTemplate =
stream.on('tweet', (tweet) => {
    console.log(`Tweet ID: ${tweet.id}`);
    tweetToAnalyse = tweet.text;
    T.post('statuses/update', 
        { status:  `I need to reply to your Tweet ID ${tweet.id}, @${tweet.user.screen_name}!`,
          in_reply_to_status_id: tweet.id_str}, 
        (err, data) => {
            if (err) {
                console.log(err);
            } else {
                console.log('Responded to call to action.');
            }
    });
});

// Making a call to Python Backend to analyse the Tweet
// Includes Topic analysis and Sentiment analysis
var topic_analyser = spawn('python', ['python-backend/topic_analyser.py']);
var topicAnalysisLog = []
var topicAnalysisResult = ''
// Gets called when data is passed out of the Python subprocess
topic_analyser.stdout.on('data', (python_output) => {
    // add any output to a results array
    topicAnalysisLog.push(python_output.toString());
    if(python_output.includes("topic=")) {
        topicAnalysisResult = python_output;
    }
    console.log(`Streamed Python output: ${python_output}`);
});
// Gets called on the end of the subprocess
topic_analyser.stdout.on('end', () => {
    console.log(`Topic analysis result = ${topicAnalysisResult}`);
});
topic_analyser.stderr.on('data', (err) => {
    console.log(`Error occured in Python subprocess!:\n${err}`);
});

topic_analyser.stdin.write(tweetToAnalyse);
topic_analyser.stdin.end();

var sentiment_analyser = spawn('python', ['python-backend/sentiment_analyser.py']);
var sentimentAnalysisLog = []
var sentimentAnalysisResult = ''
// Gets called when data is passed out of the Python subprocess
sentiment_analyser.stdout.on('data', (python_output) => {
    // add any output to a results array
    sentimentAnalysisLog.push(python_output.toString());
    if(python_output.includes("sentiment=")) {
        sentimentAnalysisResult = python_output;
    }
    console.log(`Streamed Python output: ${python_output}`);
});
// Gets called on the end of the subprocess
sentiment_analyser.stdout.on('end', () => {
    console.log(`Sentiment analysis result = ${sentimentAnalysisResult}`);
});
sentiment_analyser.stderr.on('data', (err) => {
    console.log(`Error occured in Python subprocess!:\n${err}`);
});

sentiment_analyser.stdin.write(tweetToAnalyse);
sentiment_analyser.stdin.end();

