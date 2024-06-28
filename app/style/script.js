document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed");

    const images = document.querySelectorAll('.image-item');
    console.log("Found images:", images);

    images.forEach(image => {
        image.addEventListener('click', () => {
            const imageNumber = image.getAttribute('data-image').split(' ')[1]; // Extract number from data-image attribute
            console.log("Image item clicked:", imageNumber);
            selectImage(imageNumber);
        });
    });

    const promptInput = document.getElementById('prompt');
    promptInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            editImage();
        }
    });
});
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed");

    const images = document.querySelectorAll('.image-item');
    console.log("Found images:", images);

    images.forEach(image => {
        image.addEventListener('click', () => {
            const imageNumber = image.getAttribute('data-image').split(' ')[1]; // Extract number from data-image attribute
            console.log("Image item clicked:", imageNumber);
            selectImage(imageNumber);
        });
    });

    const promptInput = document.getElementById('prompt');
    promptInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            editImage();
        }
    });
});

function selectImage(imageNumber) {
    console.log("Selected image number:", imageNumber); // Print the imageNumber for debugging
    const imagePath = `/test_images/${imageNumber}.jpg`;
    const selectedImageElement = document.getElementById('selected-image-display');
    console.log("Image path:", imagePath); // Debug the image path
    if (selectedImageElement) {
        console.log("Found selected-image-display element");
        selectedImageElement.src = imagePath;
        console.log("Image source set to:", imagePath);
    } else {
        console.error('Element with ID "selected-image-display" not found');
    }
}

async function editImage() {
    const selectedImageElement = document.getElementById('selected-image-display');
    const imageName = selectedImageElement.src.split('/').pop();
    const prompt = document.getElementById('prompt').value;

    if (!imageName || prompt === "") {
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
function selectImage(imageNumber) {
    console.log("Selected image number:", imageNumber); // Print the imageNumber for debugging
    const imagePath = `/test_images/${imageNumber}.jpg`;
    const selectedImageElement = document.getElementById('selected-image-display');
    console.log("Image path:", imagePath); // Debug the image path
    if (selectedImageElement) {
        console.log("Found selected-image-display element");
        selectedImageElement.src = imagePath;
        console.log("Image source set to:", imagePath);
    } else {
        console.error('Element with ID "selected-image-display" not found');
    }
}

async function editImage() {
    const selectedImageElement = document.getElementById('selected-image-display');
    const imageName = selectedImageElement.src.split('/').pop();
    const prompt = document.getElementById('prompt').value;

    if (!imageName || prompt === "") {
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
