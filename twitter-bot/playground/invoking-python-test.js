// Method One

var spawn = require('child_process').spawn;
var pythonScript = spawn('python', ['invoking_python_test.py']);
var input = [11,12,13,14,15];
var input2 = [11,12,13,14,16];
var resultString = '';

// Gets called when data is passed out of the Python subprocess
pythonScript.stdout.on('data', (result) => {
    resultString += result.toString();
    console.log(`Data: ${resultString}`);
});
// Gets called on the end of the subprocess
pythonScript.stdout.on('end', () => {
    console.log(`Sum of numbers = ${resultString}`);
});
pythonScript.stderr.on('data', (err) => {
    console.log(`Error occured in Python subprocess!:\n${err}`);
});

pyinput = JSON.stringify(input) + '\nTest'

pythonScript.stdin.write(pyinput);
pythonScript.stdin.end();
