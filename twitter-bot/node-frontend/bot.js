// const Seed = require('seed-text');
// const spawn = require('child_process').spawn;

const fs = require('fs');
const request = require('request');
const Twit = require('twit');
const seedtext = require('seedtext');
//Getting twitter configuration from auth.js file
const auth = require('./auth.js');
// Twitter Bot object to connect to Twitter API
const T = new Twit(auth);

const localhost = 'http://127.0.0.1:5000/'


// Preparing Seed-related stuff 
function loadSketchFromFile(path) {
    var loadedSketch = fs.readFileSync(path, 'utf-8');
    return loadedSketch;
}; 

seedSketch = loadSketchFromFile('seed/twitter_bot_seed_template.txt');   

async function generate(seedSketch, loadSketch, conditional_variables, callback) {
    const phraseBook = await seedtext.parsePhraseBook(seedSketch, loadSketch, conditional_variables);
    const generatedString = await seedtext.generateString(phraseBook, seed=42);
    callback(generatedString);
}

// Conecting to Twitter and listening for calls
var stream = T.stream('statuses/filter', { track: ['#DeescalationBot']});

// Listen for a call to action and respond to it in thread
var botResponse = '';

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
        let probabilities = JSON.parse(body.probabilities);
        let highestPrediction = undefined;
        let highestPredictionProbability = 0;

        for (let i = 0; i < body.predictions.length; i++) {
            if (probabilities[i] > highestPredictionProbability) {
                highestPrediction = body.predictions[i];
                highestPredictionProbability = probabilities[i];
            }
        }

        console.log(`Topic has been analysed as '${highestPrediction}' with probability ${highestPredictionProbability}.`);
    }
});

request({
    method: 'POST',
    url: localhost + 'analyse/anger',
    body: requestBody,
    json: true
}, (error, response, body) => {
    if (error) {
        console.log('Cannot use Python backend.');
    } else if (response.statusCode !== 200) {
        console.log(`Something went wrong with fetching response. Status code: ${response.statusCode}`);
    } else {
        console.log(`Anger has been analysed as: ${body.prediction}`);
    }
});


// stream.on('tweet', (tweet) => {
//     console.log(`Tweet ID: ${tweet.id}`);
//     tweetToAnalyse = tweet.text;
//     requestBody = { 'text': tweetToAnalyse}
//     request({
//         method: 'POST',
//         url: localhost + 'analyse/topic',
//         body: requestBody,
//         json: true
//     }, (error, response, body) => {
//         if (error) {
//             console.log('Cannot use Python backend.');
//         } else if (response.statusCode !== 200) {
//             console.log(`Something went wrong with fetching response. Status code: ${response.statusCode}`);
//         } else {
//             console.log(`Topic has been analysed as: ${JSON.stringify(body)}`);
//         }
//     });
//     setTimeout(function () {
//         T.post('statuses/update', 
//         { status:  `@${tweet.user.screen_name}: ${botResponse}`,
//         in_reply_to_status_id: tweet.id_str}, 
//         (err, data) => {
//             if (err) {
//                 console.log(err);
//             } else {
//                 console.log('Bot response: ' + botResponse);
//             }
//         });
//     }, 2000);
// });

////////////

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

