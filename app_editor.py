import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Function to add text to the image
def add_text_to_image(image, text, font_size=0.001):
    draw = ImageDraw.Draw(image)
    # Use the default PIL font
    font = ImageFont.load_default()
    # Position for the text
    text_position = (10, 10)  # You can change the position as needed
    # Add text to image
    draw.text(text_position, text, font=font, fill="white", fontsize=font_size)
    return image

# Title of the app
st.title("Image Editor App")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Adding text input for editing
    text_input = st.text_input("Enter text to overlay on image")
    # Adding font size input for editing
    # font_size = st.slider("Select font size", 10, 100, 50)

    # Add a submit button to trigger the editing
    if st.button('Submit'):
        st.write("Editing Image... Please wait.")
        edited_image = add_text_to_image(image.copy(), text_input)
        st.success("Image editing successful!")
        st.image(edited_image, caption='Edited Image with Text', use_column_width=True)

        # Add a button to save the edited image
        if st.button('Save Image'):
            edited_image.save("edited_image_with_text.png")
            st.write("Image saved successfully!")

        # Allow further editing
        image = edited_image
else:
    st.write("Please upload an image to get started.")


