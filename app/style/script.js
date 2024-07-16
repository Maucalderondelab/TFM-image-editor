document.addEventListener('DOMContentLoaded', () => {
    const imageGallery = document.getElementById('image-gallery');
    const selectedImageDisplay = document.getElementById('selected-image-display');
    const editForm = document.getElementById('edit-form');
    const selectedImageNameInput = document.getElementById('selected-image-name');
    const editButton = document.getElementById('edit-button');
    const saveEditedImageButton = document.getElementById('save-edited-image-button');
    const editedImageDisplay = document.getElementById('edited-image-display');
    const errorMessage = document.getElementById('error-message');
    const editedImageContainer = document.getElementById('edited-image-container');
    const promptInput = document.getElementById('prompt-input');

    function selectImage(image) {
        selectedImageDisplay.src = `/test_images/${image}`;
        selectedImageDisplay.alt = image;
        selectedImageNameInput.value = image;
        updateEditButtonState();
        editedImageDisplay.style.display = 'none';
        saveEditedImageButton.disabled = true;
    }

    function updateEditButtonState() {
        editButton.disabled = !promptInput.value.trim() || !selectedImageNameInput.value;
    }

    function fetchImages() {
        fetch('/images')
            .then(response => response.json())
            .then(images => {
                imageGallery.innerHTML = ''; // Clear existing images
                images.forEach((image) => {
                    const imageItem = document.createElement('div');
                    imageItem.classList.add('image-item');
                    imageItem.innerHTML = `
                        <img src="/test_images/${image}" alt="${image}">
                        <div>${image}</div>
                    `;
                    imageItem.addEventListener('click', () => selectImage(image));
                    imageGallery.appendChild(imageItem);
                });
            })
            .catch(error => displayError('Error loading images. Please try again.'));
    }

    function previewEdit(event) {
        event.preventDefault();
        const formData = new FormData(editForm);
        formData.append('prompt', promptInput.value);
    
        fetch('/preview-edit', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.detail || 'Unknown error');
                });
            }
            return response.json();
        })
        .then(data => {
            if (data && data.edited_image_data) {
                editedImageDisplay.src = data.edited_image_data;
                editedImageDisplay.style.display = 'block';
                editedImageContainer.style.display = 'block';
                saveEditedImageButton.disabled = false;
            } else {
                displayError('No valid edited image data received');
            }
        })
        .catch(error => displayError(`Error applying effect: ${error.message}. Please try again.`));
    }

    function saveEditedImage() {
        const formData = new FormData();
        formData.append('image_name', selectedImageNameInput.value);
        formData.append('edited_image_data', editedImageDisplay.src);

        fetch('/save-edit', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            if (data && data.saved_image_url) {
                alert('Image saved successfully!');
                // Optionally, you can update the UI to show the saved image or its path
            } else {
                displayError('Error saving the edited image');
            }
        })
        .catch(error => displayError('Error saving image. Please try again.'));
    }

    function displayError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        setTimeout(() => {
            errorMessage.style.display = 'none';
        }, 5000); // Hide error after 5 seconds
    }

    // Event listeners
    fetchImages(); // Load images when the page loads
    editForm.addEventListener('submit', previewEdit);
    saveEditedImageButton.addEventListener('click', saveEditedImage);
    promptInput.addEventListener('input', updateEditButtonState);

    // Initial state
    updateEditButtonState();

    // Optional: Refresh images periodically
    setInterval(fetchImages, 60000); // Refresh every minute
});