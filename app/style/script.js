function selectImage(imageNumber) {
    const imagePath = `/test_images/${imageNumber}.jpg`;
    const selectedImageElement = document.getElementById('selected-image-display');
    selectedImageElement.src = imagePath;
}


async function editImage() {
    const imageName = document.getElementById('selected-image').innerText;
    const prompt = document.getElementById('prompt').value;

    if (imageName === "Image Selected" || prompt === "") {
        alert("Please select an image and enter a prompt");
        return;
    }

    try {
        const response = await fetch('/edit-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ imageName, prompt })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const editedImages = await response.json();

        document.getElementById('e-image-1').innerText = editedImages[0];
        document.getElementById('e-image-2').innerText = editedImages[1];
        document.getElementById('e-image-3').innerText = editedImages[2];
        document.getElementById('e-image-4').innerText = editedImages[3];
    } catch (error) {
        console.error('Error editing image:', error);
        alert('Failed to edit image. Please try again.');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const images = document.querySelectorAll('.image-item');
    const selectedImageDiv = document.getElementById('selected-image');
    const promptInput = document.getElementById('prompt');

    images.forEach(image => {
        image.addEventListener('click', () => {
            selectedImageDiv.textContent = image.textContent;
        });
    });

    promptInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            editImage();
        }
    });
});
