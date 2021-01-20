import streamlit as st 
import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image


def load_image(img):
    #img = tf.io.read_file(img_path)
    #img = tf.image.decode_image(img, channels=3)
    #img = tf.image.convert_image_dtype(img, tf.float32)
    
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
style_input_method = st.sidebar.selectbox('Choose Input Method',['Webcam Click','File Upload'],1)

st.sidebar.subheader('Image Input')
img_input_method = st.sidebar.selectbox('Choose Input Method',['Webcam Stream','Webcam Click','File Upload'],2)
file_up = st.empty()
transfer = st.empty()
dp_img = st.empty()

if style_input_method == 'File Upload' :

    style_file_loc= file_up.file_uploader('Upload Your Art Work',['png','jpeg','jpg'])
    if style_file_loc:

        img = Image.open(style_file_loc)
        col1, col2 = st.beta_columns([.5,1])
        dp_img.image(img,caption='Style',use_column_width=True,width=250,height=250)


        if img_input_method == 'Webcam Stream':
            
            style_image = load_image(np.asarray(np.array(img)).astype('float32')/255)
            
            s = col1.button('Start')
            sto = col2.button('Stop')
            cap = cv2.VideoCapture(0)
         
            
            
            
            

            while s:
                _, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                content_image=load_image(np.asarray(np.array(frame)).astype('float32')/255)


                stylized_image = np.squeeze(model(tf.constant(content_image), tf.constant(style_image))[0])
                dp_img.image(stylized_image,use_column_width=True,width=250,height=250)

            if sto:
                dp_img = st.empty()
                cap.release()
                st.write('Stopped')






        if img_input_method == 'File Upload':
            img_file_loc =file_up.file_uploader('Upload Your Input Imae Work',['png','jpeg','jpg'])
            if img_file_loc:
                img2 = Image.open(img_file_loc)
                dp_img.image(img2,caption='Input',use_column_width=True,width=250,height=250)
                transfer.button("Transfer Style")
                if transfer:
                    
                    content_image = load_image(np.asarray(np.array(img2)).astype('float32')/255)
                    style_image = load_image(np.asarray(np.array(img)).astype('float32')/255)


                    stylized_image = np.squeeze(model(tf.constant(content_image), tf.constant(style_image))[0])
                    dp_img.image(stylized_image,use_column_width=True,width=250,height=250)

