document.addEventListener('DOMContentLoaded', () => {
    const imageGallery = document.getElementById('image-gallery');
    const selectedImageDisplay = document.getElementById('selected-image-display');
    const editForm = document.getElementById('edit-form');
    const selectedImageNameInput = document.getElementById('selected-image-name');
    const applyEffectButton = document.getElementById('apply-effect-button');
    const editedResults = document.getElementById('edited-results');


    // Function to handle image selection
    function selectImage(image) {
        selectedImageDisplay.src = `/test_images/${image}`;
        selectedImageDisplay.alt = image;
        selectedImageNameInput.value = image;
        editButton.disabled = false;

    }
    
    // Function to fetch and display images
    function fetchImages() {
        fetch('/images')
            .then(response => response.json())
            .then(images => {
                const imageGallery = document.getElementById('image-gallery');
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
            .catch(error => {
                console.error('Error applying effect:', error);
                editedResults.innerHTML = `<p>Error: ${error.message}</p>`;
            });
    }

    // Function to apply black and white effect
    function applyBlackAndWhiteEffect(event) {
        event.preventDefault();
        const formData = new FormData(editForm);
    
        fetch('/edit-image', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            const editedImageUrl = data.edited_image_url;
            editedResults.innerHTML = '';  // Clear previous edited image
            const editedItem = document.createElement('div');
            editedItem.classList.add('edited-item');
            editedItem.innerHTML = `<img src="${editedImageUrl}" alt="Edited Image">`;
            editedResults.appendChild(editedItem);
            
            // Update the selected image display with the edited image
            selectedImageDisplay.src = editedImageUrl;
        })
        .catch(error => {
            console.error('Error applying effect:', error);
            editedResults.innerHTML = `<p>Error: ${error.message}</p>`;
        });
    }

    // Fetch and display images on page load
    fetchImages();

    // Event listener for the edit form submission
    editForm.addEventListener('submit', applyBlackAndWhiteEffect);
});