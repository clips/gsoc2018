# Twitter Bot Frontend (Node.js)

## Overview

There are two important parts of the frontend. First is the bot itself, the second is the ensemble of seed sources which provide response generation capabilities. The bot is almost entirely to be found in the _bot.js_ file. The seed sources are in the */seedtext* directory.

## Bot 

The source code for combines many different aspects. Firstly, it provides access to Python backend using microservices via calls to the Flask server.

```javascript
request({
        method: 'POST',
        url: localhost + 'analyse/topic',
        body: requestBody,
        json: true
    }, (error, response, body) => {
    	// CODE
    }
});
```

Secondly, the Twitter API is accessed using the [Twit module](https://www.npmjs.com/package/twit).

```javascript
stream.on('tweet', (tweet) => {
    analysedTweet.tweetID = tweet.id_str;
    console.log(`Tweet ID: ${analysedTweet.tweetID}`);
    analysedTweet.tweetText = tweet.text.replace('#DeescalationBot', '');
    console.log(`Tweet Text: ${analysedTweet.tweetText }`);
    analysedTweet.tweetUser = tweet.user.name;
    requestBody = { 'text': analysedTweet.tweetText }
});
```

And last, but not least, it makes calls into the [seedtext module](https://www.npmjs.com/package/seedtext) that generate the responses based on a number of factors that were analysed.

```javascript
function loadSketchFromFile(path) {
    var loadedSketch = fs.readFileSync(path, 'utf-8');
    return loadedSketch;
}; 

seedSketch = loadSketchFromFile('seed/bot.txt');   

async function generate(seedSketch, loadSketch, conditional_variables, globalMemory, callback) {
    const phraseBook = await seedtext.parsePhraseBook(seedSketch, loadSketch, conditional_variables);
    const generatedString = await seedtext.generateString(phraseBook, 'root', globalMemory, seed=42);
    callback(generatedString);
}

generate(seedSketch, loadSketchFromFile, conditionalVars, {}, (result) => {
	// CODE TO SEND TWEET
});
```

This brings us to the second part.

## SeedText Generator

In the directory */seed* there is a number of Seed source files that provide response generation capabilities. At the moment of writing, these are:

- *bot.txt*: The main source file. Imports all of the others and delegates to them if needed. Provides generation rules for non-special cases.
- *disengageMessage.txt*: Supplementary source file. used when the bot decides to disengage from discussion (discussion was calmed down or doesn't lead anywhere).
- *firstReply.txt*: Supplementary source file. It is used when first replying to a new Tweet thread.
- *unknownTopicResponse:* Supplementary source file. Handles cases when topic was predicted with unsatisfactory confidence (lower than 60% confidence).

All of these source files are read, lexed, parsed and interpreted by *seedtext* for each generated response. To see how to write your own Seed files, please, refer to the [official documentation](https://seed.emrg.be/docs).

