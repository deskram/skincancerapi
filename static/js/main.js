// main.js

document.addEventListener("DOMContentLoaded", function () {
    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-input");
    const submitButton = document.getElementById("submit-button");
    const resultDiv = document.getElementById("result");
    const predictionSpan = document.getElementById("prediction");
    const predictedImage = document.getElementById("predicted-image");

    uploadForm.addEventListener("submit", function (e) {
        e.preventDefault();
        submitButton.disabled = true;

        const formData = new FormData(uploadForm);
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            predictionSpan.textContent = data.prediction;

            // تعيين مسار الصورة بشكل صحيح باستخدام url_for
            predictedImage.src = data.image_path + '?' + new Date().getTime();

            resultDiv.style.display = "block";
            submitButton.disabled = false;
        })
        .catch(error => {
            console.error('Error:', error);
            submitButton.disabled = false;
        });
    });
    
    fileInput.addEventListener("change", function () {
        submitButton.disabled = !fileInput.files.length;
    });
});
