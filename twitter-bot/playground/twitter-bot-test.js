var Twitter = require('twitter')
//Authentication info from auth.js file
var auth = require('./auth.js')

// initialize authenticated Twitter instance
var T = new Twitter(auth);

var params = {
    q: 'vegan',
    count: 10,
    result_type: 'recent',
    lang: 'en'
}

// Code to favourite all returned Tweets
T.get('search/tweets', params, function(err, data, response) {
    if(!err){
        for(let i = 0; i < data.statuses.length; i++){
            console.log(`Status ${i}: ${data.statuses[i].tweet}`)
            // Get the tweet Id from the returned data
            let id = { id: data.statuses[i].id_str }
            // Try to Favorite the selected Tweet
            T.post('favorites/create', id, function(err, response){
            // If the favorite fails, log the error message
            if(err){
                console.log(`Error: ${err[0].message}`);
            }
            // If the favorite is successful, log the url of the tweet
            else{
                let username = response.user.screen_name;
                let tweetId = response.id_str;
                console.log('Favorited: ', `https://twitter.com/${username}/status/${tweetId}`)
            }
            });
        }
    } else {
      console.log(err);
    }
  });


// //Twitter object to connect to Twitter API
// var T = new Twit(auth);
// var stream = T.stream('user');
// //listens to the event when someone follows and calls 
// //callback function followed 
// stream.on('follow', followed);

// function followed(eventmsg) {
// 	//getting name and username of the user
//     var name = eventmsg.source.name;
//     var screenName = eventmsg.source.screen_name;
//     //since twitter blocks tweets of same type so we'll associate a
//     //unique number using Math.random() or anything you like
//     tweetPost('.@' + screenName + 'Yay you followed me! Your lucky number:' + Math.floor(Math.random()*10));
// }
// //Posting the tweet!
// function tweetPost(msg) {
//     var tweet = {
//         status: msg
//     }
//     T.post('statuses/update', tweet, function(err, data) {
//         if (err) {
//             console.log(err);
//         } else {
//             console.log(data);
//         }
//     });
// }