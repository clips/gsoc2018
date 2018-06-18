// const Seed = require('seed-text');
// const spawn = require('child_process').spawn;
const request = require('request')
const Twit = require('twit')
//Getting twitter configuration from auth.js file
const auth = require('./auth.js')
// Twitter Bot object to connect to Twitter API
const T = new Twit(auth);

const localhost = 'http://127.0.0.1:5000/'

var stream = T.stream('statuses/filter', { track: ['#DeescalationBot']});
var tweetToAnalyse = 'This is a sample angry tweet about president';

// Listen for a call to action and respond in thread to it.
var botResponse = 'Placeholder';

stream.on('tweet', (tweet) => {
    console.log(`Tweet ID: ${tweet.id}`);
    tweetToAnalyse = tweet.text;
    requestBody = { 'method': 'nmf', 'text': tweetToAnalyse}
    request({
        method: 'POST',
        url: localhost + 'analyse/topic',
        body: requestBody,
        json: true
    }, (error, response, body) => {
        if (error) {
            console.log('Cannot use Python backend.');
        } else if (response.statusCode !== 200) {
            console.log(`Something went wrong with fetching response. Status code: ${response.statusCode}`);
        } else {
            // WILL BE OUTSOURCED FROM SEED
            if (body.topics === 'something') {
                responseTemplates = [
                    'I am sorry but I do not understand. Could you elaborate? Thanks.',
                    'I apologize but I am not sure whether I understand you correctly. Could you tell me more about the subject?'
                ]
                randIndex = Math.floor(Math.random() * Math.floor(responseTemplates.length));
                botResponse = responseTemplates[randIndex];
            } else {
                responseTemplates = [
                    'I can see that your are unhappy about ' +  body.topics + '. Could you help me understand why?',
                    'I understand. So you are saying that ' +  body.topics + ' makes you feel angry. Could you tell me more about it?',
                    'So, if I understand it correctly, it is ' +  body.topics + ' that makes you feel upset, is that correct?',
                    'I see. So you are feeling bad because of the situation with ' + body.topics + ', do I understand that correctly?',
                    'So you are feeling bad because of the situation with ' + body.topics + ', do I understand that correctly?',
                    'I think I know what you mean. You are unhappy about ' + body.topics + ', is that correct?',
                    'I hear you. So you want to say that ' + body.topics + ' makes you feel upset?'
                ]
                randIndex = Math.floor(Math.random() * Math.floor(responseTemplates.length));
                botResponse = responseTemplates[randIndex];
            }
        }
    });
    setTimeout(function () {
        T.post('statuses/update', 
        { status:  `@${tweet.user.screen_name}: ${botResponse}`,
        in_reply_to_status_id: tweet.id_str}, 
        (err, data) => {
            if (err) {
                console.log(err);
            } else {
                console.log('Bot response: ' + botResponse);
            }
        });
    }, 2000);
});

// // Making a call to Python Backend to analyse the Tweet
// // Includes Topic analysis and Sentiment analysis
// var topic_analyser = spawn('python', ['python-backend/topic_analyser.py']);
// var topicAnalysisLog = []
// var topicAnalysisResult = ''
// // Gets called when data is passed out of the Python subprocess
// topic_analyser.stdout.on('data', (python_output) => {
//     // add any output to a results array
//     topicAnalysisLog.push(python_output.toString());
//     if(python_output.includes("topic=")) {
//         topicAnalysisResult = python_output;
//     }
//     console.log(`Streamed Python output: ${python_output}`);
// });
// // Gets called on the end of the subprocess
// topic_analyser.stdout.on('end', () => {
//     console.log(`Topic analysis result = ${topicAnalysisResult}`);
// });
// topic_analyser.stderr.on('data', (err) => {
//     console.log(`Error occured in Python subprocess!:\n${err}`);
// });

// topic_analyser.stdin.write(tweetToAnalyse);
// topic_analyser.stdin.end();

// var sentiment_analyser = spawn('python', ['python-backend/sentiment_analyser.py']);
// var sentimentAnalysisLog = []
// var sentimentAnalysisResult = ''
// // Gets called when data is passed out of the Python subprocess
// sentiment_analyser.stdout.on('data', (python_output) => {
//     // add any output to a results array
//     sentimentAnalysisLog.push(python_output.toString());
//     if(python_output.includes("sentiment=")) {
//         sentimentAnalysisResult = python_output;
//     }
//     console.log(`Streamed Python output: ${python_output}`);
// });
// // Gets called on the end of the subprocess
// sentiment_analyser.stdout.on('end', () => {
//     console.log(`Sentiment analysis result = ${sentimentAnalysisResult}`);
// });
// sentiment_analyser.stderr.on('data', (err) => {
//     console.log(`Error occured in Python subprocess!:\n${err}`);
// });

// sentiment_analyser.stdin.write(tweetToAnalyse);
// sentiment_analyser.stdin.end();

