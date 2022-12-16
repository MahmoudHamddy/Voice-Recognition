record_button = document.getElementById("recording_button")

record_button.onclick = ()=>{
    record_button.style.animationName = "pulsating";
    setTimeout(() => record_button.style.animationName = "", 3000);
}
