var request = require('request');

var geocodeAddress = (address) => {

};

geocodeAddress('26 albion street hu13tb').then((location) => {
    console.log(JSON.stringify(location, undefined, 2));
}, (errorMessage) => {
    console.log(errorMessage);
})