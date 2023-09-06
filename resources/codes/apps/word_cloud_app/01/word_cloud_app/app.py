# Required installations:
# pip install streamlit wordcloud pandas python-docx pdfminer.six matplotlib
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from pdfminer.high_level import extract_text
import io

st.set_option('deprecation.showPyplotGlobalUse', False)

def read_txt(file):
    return file.getvalue().decode("utf-8")

def read_docx(file):
    doc = Document(io.BytesIO(file.read()))
    return ' '.join([p.text for p in doc.paragraphs])

def read_pdf(file):
    return extract_text(io.BytesIO(file.read()))

st.title("Sadi Word Cloud di app")

uploaded_files = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx'], accept_multiple_files=True)

# Sidebar for customization
st.sidebar.header("Customization Options")
width = st.sidebar.slider("Width of Word Cloud", 400, 2000, 800)
height = st.sidebar.slider("Height of Word Cloud", 400, 2000, 800)
resolution = st.sidebar.slider("Resolution (DPI)", 50, 300, 100)
formats = ["PNG", "JPEG", "SVG"]
file_format = st.sidebar.selectbox("Select File Format", formats)

if uploaded_files:
    combined_text = ''
    for file in uploaded_files:
        if '.txt' in file.name:
            combined_text += read_txt(file)
        elif '.docx' in file.name:
            combined_text += read_docx(file)
        elif '.pdf' in file.name:
            combined_text += read_pdf(file)

    stop_words = set(STOPWORDS)

    if st.checkbox("Remove Stopwords"):
        wordcloud = WordCloud(stopwords=stop_words, width=width, height=height).generate(combined_text)
    else:
        wordcloud = WordCloud(width=width, height=height).generate(combined_text)

    word_freq = pd.DataFrame(wordcloud.process_text(combined_text).items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)

    additional_stop_words = st.multiselect("Select additional stopwords:", word_freq["Word"][:50].tolist())
    
    if additional_stop_words:
        stop_words.update(additional_stop_words)
        wordcloud = WordCloud(stopwords=stop_words, width=width, height=height).generate(combined_text)
    
    plt.figure(figsize=(width/100, height/100), dpi=resolution)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot()

    st.write("Word Frequencies")
    st.table(word_freq.head(10))

    if st.button('Download Word Cloud'):
        buffered = io.BytesIO()
        if file_format == "PNG":
            plt.savefig(buffered, format="PNG", dpi=resolution)
            st.download_button(
                label="Download Word Cloud as PNG",
                data=buffered,
                file_name="word_cloud.png",
                mime="image/png"
            )
        elif file_format == "JPEG":
            plt.savefig(buffered, format="JPEG", dpi=resolution)
            st.download_button(
                label="Download Word Cloud as JPEG",
                data=buffered,
                file_name="word_cloud.jpeg",
                mime="image/jpeg"
            )
        elif file_format == "SVG":
            buffered_svg = io.StringIO()
            plt.savefig(buffered_svg, format="SVG", dpi=resolution)
            st.download_button(
                label="Download Word Cloud as SVG",
                data=buffered_svg.getvalue(),
                file_name="word_cloud.svg",
                mime="image/svg+xml"
            )

    st.write("Connect with me:")
    social_media = {
        "LinkedIn": "",
        "Twitter": "https://twitter.com/aammar_tufail",
        "Instagram": ""
    }
    
    for media, link in social_media.items():
        st.write(f"{media}: {link}")

# Required installations:
# pip install streamlit wordcloud pandas python-docx pdfminer.six matplotlib

