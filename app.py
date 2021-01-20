import streamlit as st 
import tensorflow as tf
from PIL import Image


def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

@st.cache()
def load_model():
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    return model
model = load_model()


st.title('Make you own filter')
st.subheader('Bring artwork to life')

st.sidebar.header('Choose Your Input Partaneters')
st.sidebar.subheader('Style Input')
style_input_method = st.sidebar.selectbox('Choose Input Method',['Webcam Click','File Upload'])

st.sidebar.subheader('Image Input')
img_input_method = st.sidebar.selectbox('Choose Input Method',['Webcam Stream','Webcam Click','File Upload'])
file_up = st.empty()
dp_img = st.empty()

if style_input_method == 'File Upload' :

    style_file_loc= file_up.file_uploader('Upload Your Art Work',['png','jpeg','jpg'])
    if style_file_loc:

        img = Image.open(style_file_loc)
        dp_img.image(img,caption='Style',use_column_width=True,width=250,height=250)

        if img_input_method == 'File Upload':
            img_file_loc =file_up.file_uploader('Upload Your Input Imae Work',['png','jpeg','jpg'])
            if img_file_loc:
                img2 = Image.open(style_file_loc)
                dp_img.image(img2,caption='Input',use_column_width=True,width=250,height=250)
    