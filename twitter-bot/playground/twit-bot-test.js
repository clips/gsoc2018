var Twit = require('twit')
//Getting twitter configuration from auth.js file
var auth = require('./auth.js')
//Twitter object to connect to Twitter API
var T = new Twit(auth);
var stream = T.stream('user');
var sessionFollowSum = 0;

//listens to the event when someone follows and calls 
//callback function followed 
stream.on('follow', followed);

function followed(eventmsg) {
	//getting name and username of the user
    var name = eventmsg.source.name;
    var screenName = eventmsg.source.screen_name;

    sessionFollowSum += 1;
    tweetPost(`Thanks for following me, @${screenName}! Total of ${sessionFollowSum} did this session.`);
}


function tweetPost(msg) {
    var tweet = {
        status: msg
    }
    T.post('statuses/update', tweet, function(err, data) {
        if (err) {
            console.log(err);
        } else {
            console.log(data);
        }
    });
}