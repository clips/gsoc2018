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

seedSketch = loadSketchFromFile('testseed.txt');   

async function generate(seedSketch, loadSketch, conditional_variables, globalMemory, callback) {
    const phraseBook = await seedtext.parsePhraseBook(seedSketch, loadSketch, conditional_variables);
    const generatedString = await seedtext.generateString(phraseBook, 'root', globalMemory, seed=42);
    callback(generatedString);
}

// Conecting to Twitter and listening for calls
var stream = T.stream('statuses/filter', { track: ['#DeescalationBot']});

// Listen for a call to action and respond to it in thread
var botResponse = '';
var analysedTweet = {
    'tweetID':'',
    'tweetUser':'',
    'tweetText':'',
    'analysedTopic':'',
    'analysedTopicConfidence':'',
    'analysedAnger':'',
}

stream.on('tweet', (tweet) => {
    analysedTweet.tweetID = tweet.id_str;
    console.log(`Tweet ID: ${analysedTweet.tweetID}`);
    analysedTweet.tweetText = tweet.text.replace('#DeescalationBot', '');
    console.log(`Tweet Text: ${analysedTweet.tweetText }`);
    analysedTweet.tweetUser = tweet.user.name;
    requestBody = { 'text': analysedTweet.tweetText }
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

            analysedTweet.analysedTopic = highestPrediction;
            analysedTweet.analysedTopicConfidence = highestPredictionProbability;

            // a bit of a dirty hack for now - storing topic as a seed source that will be imported
            topicSeedProgram = 'root:\n- ' + analysedTweet.analysedTopic.replace(/['"]+/g, '');
            fs.writeFileSync('topic.txt',  topicSeedProgram);

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
                    anger = body.prediction;
                    if (anger < 0.001) {
                        anger = 0;
                    }
                    console.log(`Anger has been analysed as: ${anger}`);

                    analysedTweet.analysedAnger = anger;

                    generate(seedSketch, loadSketchFromFile, {'anger': analysedTweet.analysedAnger}, {}, (result) => {
                        botResponse = result;
                        T.post('statuses/update', 
                        { status:  `${botResponse}`,
                        in_reply_to_status_id: analysedTweet.tweetID}, 
                        (err, data) => {
                            if (err) {
                                console.log(err);
                            } else {
                                console.log('Bot response: ' + botResponse);
                            }
                        });
                    });
                }
            });
        }
    });
});
