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

requestBody = { 'text': 'This tweet talks about European Union in a non-agry way'};
console.log(`The tweet to analyse is: ${requestBody.text}`);
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

        // console.log(probabilities[0]);

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
        anger = body.prediction;
        if (anger < 0.001) {
            anger = 0;
        }
        console.log(`Anger has been analysed as: ${anger}`);
    }
});

