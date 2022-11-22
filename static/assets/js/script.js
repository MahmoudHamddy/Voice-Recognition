record_button = document.getElementById("recording_button");
text_recorded = document.getElementById("text_recorded");
said_text = document.getElementById("said_text");
door = document.querySelector(".door");

var isDoorOpen = false;

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();
recognition.lang = 'en-US';

record_button.onclick = ()=>{
    record_button.style.animationName = "pulsating";
    recognition.start();
}

recognition.onresult = (event) => {
    const text = Array.from(event.results)
    .map(result=>result[0])
    .map(result => result.transcript)
    .join("");
    text_recorded.innerText = text.toUpperCase()
    

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

