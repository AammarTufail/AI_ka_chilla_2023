import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO
import base64

st.set_page_config(layout="wide", page_title="Image Background Remover")

st.write("## Remove background from your image")
st.write(
    "Try uploading an image to watch the background magically removed. Full-quality images can be downloaded from the sidebar. "
)
st.sidebar.write("## Upload and download :gear:")


# Include sidebar with credentials
with st.sidebar:
    st.markdown('Clean image background (V 0.1)')
    st.markdown(""" 
                #### Let's connect:
                [Kamran Feroz](https://www.linkedin.com/in/kamranferoz/)

                #### Powered by:
                [Langchain](https://github.com/hwchase17/langchain)\n

                #### Source code:
                [Clean Image!](https://github.com/kamranferoz/bgRemoval)
                """)
st.markdown(
    "<style>#MainMenu{visibility:hidden;}</style>",
    unsafe_allow_html=True)


MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    fixed = remove(image)
    col2.write("Fixed Image :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
else:
    fix_image("./kf2.png")
