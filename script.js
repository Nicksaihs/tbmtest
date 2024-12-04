let model;
const video = document.getElementById('camera');
const resultDiv = document.getElementById('result');
const scanBtn = document.getElementById('scan');

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
        audio: false
    });
    video.srcObject = stream;
    await new Promise(resolve => (video.onloadedmetadata = resolve));
    video.play();
}

async function loadModel() {
    model = await tf.loadLayersModel('./model/model.json');
    console.log("Model loaded");
}

async function predict() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const image = tf.browser.fromPixels(canvas)
        .resizeNearestNeighbor([224, 224])
        .expandDims(0)
        .toFloat()
        .div(tf.scalar(255));
    const predictions = await model.predict(image).data();
    const labels = ["SpongeBob", "Patrick", "Squidward"];
    const maxIdx = predictions.indexOf(Math.max(...predictions));
    resultDiv.textContent = `Result: ${labels[maxIdx]}`;
}

scanBtn.addEventListener('click', predict);

setupCamera();
loadModel();
