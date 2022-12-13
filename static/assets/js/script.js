record_button = document.getElementById("recording_button");
text_recorded = document.getElementById("text_recorded");
whose_recorded = document.getElementById("whose_recorded");
said_text = document.getElementById("said_text");
door = document.querySelector(".door");

var isDoorOpen = false;


const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();
recognition.lang = 'en-US';
recognition.continuous = false;

$.ajax({
    url: "test.html",
    cache: false,
    success: function(html){
      $("#results").append(html);
    }
  });

record_button.onclick = ()=>{
    // record_button.style.color = 'green';
    // record_button.classList.remove("btn-primary")
    // record_button.classList.remove("btn")
    record_button.classList.className=".rec";
    record_button.style.animationName = "pulsating";
    recognition.start();
    // init();
    // recognition.onspeechend = () => {
    //     recognition.stop();
    //   }
}

recognition.onresult = (event) => {
    console.log("I'm in the JS")
    const text = Array.from(event.results)
    .map(result=>result[0])
    .map(result => result.transcript)
    .join("");
    text_recorded.innerText = "You said: " + text.toUpperCase()

    if (text.toLowerCase().includes("open the door")) {
        said_text.innerText = ""
        if(!isDoorOpen){
        door.classList.toggle("doorOpen");
        isDoorOpen = true;
        }
    }
    else if (text.toLowerCase().includes("close the door")){
        said_text.innerText = ""
        if (isDoorOpen) {
            door.classList.toggle("doorOpen");   
            isDoorOpen = false;
        }
    }
    else{
        said_text.innerText = "Incorrect Password"
    }
    record_button.style.animationName = "";
  }

const URL = "https://teachablemachine.withgoogle.com/models/g19IVvfMD/";

async function createModel() {
    const checkpointURL = URL + "model.json"; // model topology
    const metadataURL = URL + "metadata.json"; // model metadata

    const recognizer = speechCommands.create(
        "BROWSER_FFT", // fourier transform type, not useful to change
        undefined, // speech commands vocabulary feature, not useful for your models
        checkpointURL,
        metadataURL);

    // check that model and metadata are loaded via HTTPS requests.
    await recognizer.ensureModelLoaded();
    return recognizer;
}

//------------------------------------------------------------------------------------------------------
// async function init() {
//     const recognizer = await createModel();
//     const classLabels = recognizer.wordLabels(); // get class labels

//     //const classPrediction="";
//     // listen() takes two arguments:
//     // 1. A callback function that is invoked anytime a word is recognized.
//     // 2. A configuration object with adjustable fields
//     recognizer.listen(result => {
//         const scores = result.scores; // probability of prediction for each class
//         const max_score = Math.max(...scores)
//         // render the probability scores per class
//         // for (let i = 0; i < classLabels.length; i++) {
//             // if (result.scores[i].toFixed(2)>=0.77){  
//         const classPrediction = classLabels[scores.indexOf(max_score)] + ": " + max_score.toFixed(2);
//         whose_recorded.innerHTML = classPrediction;
//         setTimeout(() => recognizer.stopListening(), 1000);
//             // }    
//         // }
//     }, {
//         includeSpectrogram: true, // in case listen should return result.spectrogram
//         probabilityThreshold: 0.75,
//         invokeCallbackOnNoiseAndUnknown: true,
//         overlapFactor: 0.60 // probably want between 0.5 and 0.75. More info in README
//     });

    // Stop the recognition in 5 seconds.
    // setTimeout(() => recognizer.stopListening(), 2000);
// }

