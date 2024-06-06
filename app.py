from sklearn.preprocessing import LabelEncoder
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from simplet5 import SimpleT5
import tensorflow_hub as hub
import streamlit as st
import pandas as pd
import string
import joblib
import torch
import numpy
import ast
import re


def generate_title_from_poem(poem_text):
    # clean the poem
    punctuation_chars = set(string.punctuation)
    cleaned_string = ''.join(char for char in poem_text if char not in punctuation_chars)
    cleaned_string = cleaned_string.replace('\n', '')
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string)
    cleaned_string = cleaned_string.lower()
    # initialize model
    model = SimpleT5()
    model.load_model("t5","title_generation/simplet5-epoch-4-train-loss-nan-val-loss-2.9005", use_gpu=True)
    return model.predict(cleaned_string)[0]

def generate_poem_from_title(input_title):
    poem_title = str(input_title)
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
    messages = [
                {
                    "role": "system",
                    "content": "You are a poet who writes poetry based on the title",
                },
                {"role": "user", "content": f"write a mini poetry titled \"{poem_title}\""},
                ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    outputs = pipe(prompt, max_new_tokens=500, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    return outputs[0]["generated_text"].split("<|assistant|>\n")[1]

def detect_poem_topic(poem_text):
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = numpy.load('poem_classification\classes.npy', allow_pickle=True)
    # Load the model and tokenizer
    model = BertForSequenceClassification.from_pretrained("poem_classification\saved_model")
    tokenizer = BertTokenizer.from_pretrained("poem_classification\saved_model")

    # Tokenize the input poems
    poems = [poem_text.replace("\n", " ")]
    inputs = tokenizer(poems, padding=True, truncation=True, return_tensors="pt", max_length=128)

    # Get predictions
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)

    # Decode the predicted labels
    predicted_labels = label_encoder.inverse_transform(predictions.cpu().numpy())
    return predicted_labels[0]

def get_recommendations(favorite_poem):
    text= str(favorite_poem).replace("\n", " ")
    enc = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") # load the universal sentence encoder
    df = pd.read_pickle("poems_recommendation\data_for_recommandations.pkl") # load poems for recommandations
    nn = joblib.load('poems_recommendation\\5nn_model.joblib') # load the knn model trained on poems
    emb = enc([text])
    neighbors = nn.kneighbors(emb, return_distance=False)[0]
    return df.iloc[neighbors]

def read_poem(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().replace("\n", "<br>")
    except FileNotFoundError:
        return "Poem file not found."


if __name__ == '__main__':
    
    #####################################################################################################################
    #################################################  PAGE SET UP   ####################################################
    #####################################################################################################################

    st.set_page_config(page_title="POETIFY",
                    page_icon="poem_icon.png",
                    layout="wide",
                    initial_sidebar_state="expanded"
                    )

    def p_title(title):
        st.markdown(f'<h3 style="text-align: left; color:#F63366; font-size:28px;">{title}</h3>', unsafe_allow_html=True)

    #####################################################################################################################
    #################################################    SIDEBAR     ####################################################
    #####################################################################################################################

    # Sidebar configuration
    st.sidebar.markdown('# :red[**POETIFY, Let\'s :scroll:**]')
    # Info section
    st.sidebar.markdown(':white[**Generate, explore, and write poetry with ease. Try our tools, check our code, and share your creations with the world!**]')
    nav = st.sidebar.selectbox('', [
        '🏠 Go to homepage',
        '📝 Generate poem from title',
        '🔤 Generate title from poem',
        '🔍 Detect the poem topic',
        '📚 Get recommendation'
        ])
    
    # Contact information
    st.sidebar.markdown('## :red[Contact]')
    st.sidebar.markdown(":white[I'd love your feedback :smiley: Want to collaborate? Develop a project? Find us on [Mouad's LinkedIn](https://www.linkedin.com/in/mouad-ait-hammou-468b49249/) and [Oussama's LinkedIn](https://www.linkedin.com/in/bouguilim/), [Mouad's GitHub](https://github.com/feevemouad) and [Oussama's GitHub](https://github.com/Bouguilim)]")

    # Source code link
    st.sidebar.markdown('## :red[Source Code]')
    st.sidebar.markdown(':white[Find the source code [here](https://github.com/feevemouad/PoemNLProject)]')

    #####################################################################################################################
    #################################################   HOME PAGE    ####################################################
    #####################################################################################################################

    if nav == '🏠 Go to homepage':
        st.image("Images/im.png", width=700)
        st.markdown("<h2 style='color:#FF0000;'>What is this App about?</h2>", unsafe_allow_html=True)
        st.write("**POETIFY is your go-to tool for all things poetry. Whether you want to generate new poems, find the perfect title, or analyze themes, POETIFY has you covered.**")
        st.write("**Explore and enhance your poetic creativity. Choose a task, provide your input, and let POETIFY do the rest!**")

        st.markdown("<h2 style='color:#FF0000;'>Who is this App for?</h2>", unsafe_allow_html=True)
        st.write("**This app is for anyone with a love for poetry! It's completely free to use. If you enjoy it, share your support!**")
        st.write("**Are you into poetry and tech? Our code is 100% open source and written for easy understanding. Fork it on [GitHub](https://github.com/feevemouad/PoemNLProject), and share your suggestions. Join the community and help yourself and others!**")

    #####################################################################################################################
    #################################################   TITLEGEN     ####################################################
    #####################################################################################################################

    if nav == '🔤 Generate title from poem':
        st.markdown("<h4 style='text-align: center; color:grey;'>Discover the essence with Poetify 🌟</h4>", unsafe_allow_html=True)
        st.text('')
        st.text('')

        # Option to upload a poem as a text file
        st.markdown("<h5 style='text-align: left; color:grey;'>Upload a poem as a text file:</h5>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])

        # Text area to manually input the poem
        st.markdown("<h5 style='text-align: left; color:grey;'>Or write it below:</h5>", unsafe_allow_html=True)
        poem_example = "I wandered lonely as a cloud\nThat floats on high o'er vales and hills,\nWhen all at once I saw a crowd,\nA host, of golden daffodils;\nBeside the lake, beneath the trees,\nFluttering and dancing in the breeze."
        input_poem = st.text_area("Enter the poem", placeholder=poem_example, height=160)

        if st.button('Generate Title'):
            if uploaded_file is None and not input_poem:
                st.error('Please upload a text file or enter a poem')
            else:
                with st.spinner('Generating title...'):
                    if uploaded_file is not None:
                        # Read the uploaded file
                        poem_text = uploaded_file.read().decode("utf-8")
                    else:
                        poem_text = input_poem

                    # Code for generating title from poem goes here
                    title = generate_title_from_poem(poem_text)

                    # Display the generated title
                    st.markdown('___')
                    st.write('Generated Title:')
                    st.success(title)
                    
    #####################################################################################################################
    #################################################    POEMGEN     ####################################################
    #####################################################################################################################

    if nav == '📝 Generate poem from title':
        st.markdown("<h4 style='text-align: center; color:grey;'>Unleash creativity with Poetify 📜</h4>", unsafe_allow_html=True)
        st.text('')
        st.text('')

        title_example = "Dreams"
        input_title = st.text_input("Enter a title for your poem", placeholder="e.g., 'Dear Brother', 'My Guiding Moonlight', ...")

        if st.button('Generate Poem'):
            if not input_title:
                st.error('Please enter a title for your poem')
            else:
                with st.spinner('Generating poem...'):
                    # Code for generating poem goes here
                    poem = generate_poem_from_title(input_title)

                    # Display the generated poem
                    st.markdown('___')
                    st.write('Generated Poem:')
                    st.write(poem)

    #####################################################################################################################
    ################################################# CLASSIFICATION ####################################################
    #####################################################################################################################

    if nav == '🔍 Detect the poem topic':
        st.markdown("<h4 style='text-align: center; color:grey;'>Explore the themes with Poetify 🔍</h4>", unsafe_allow_html=True)
        st.text('')
        st.text('')

        # Option to upload a poem as a text file
        st.markdown("<h5 style='text-align: left; color:grey;'>Upload a poem as a text file:</h5>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])

        # Text area to manually input the poem
        st.markdown("<h5 style='text-align: left; color:grey;'>Or write it below:</h5>", unsafe_allow_html=True)
        poem_example = "I wandered lonely as a cloud\nThat floats on high o'er vales and hills,\nWhen all at once I saw a crowd,\nA host, of golden daffodils;\nBeside the lake, beneath the trees,\nFluttering and dancing in the breeze."
        input_poem = st.text_area("Enter the poem", placeholder=poem_example, height=160)

        if st.button('Detect Topic'):
            if uploaded_file is None and not input_poem:
                st.error('Please upload a text file or enter a poem')
            else:
                with st.spinner('Detecting topic...'):
                    if uploaded_file is not None:
                        # Read the uploaded file
                        poem_text = uploaded_file.read().decode("utf-8")
                    else:
                        poem_text = input_poem

                    # Code for detecting topic of the poem goes here
                    topic = detect_poem_topic(poem_text)

                    # Display the detected topic
                    st.markdown('___')
                    st.write('Detected Topic:')
                    st.success(topic)
                    
    #####################################################################################################################
    ################################################# RECOMMENDATION ####################################################
    #####################################################################################################################

    if nav == '📚 Get recommendation':
        st.markdown("<h4 style='text-align: center; color:grey;'>Discover new poems with Poetify 📚</h4>", unsafe_allow_html=True)
        st.text('')
        st.text('')

        # Option to upload the user's favorite poem as a text file
        st.markdown("<h5 style='text-align: left; color:grey;'>Upload your favorite poem as a text file:</h5>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])

        # Text area to manually input the user's favorite poem
        st.markdown("<h5 style='text-align: left; color:grey;'>Or write it below:</h5>", unsafe_allow_html=True)
        favorite_poem_example = "I wandered lonely as a cloud\nThat floats on high o'er vales and hills,\nWhen all at once I saw a crowd,\nA host, of golden daffodils;\nBeside the lake, beneath the trees,\nFluttering and dancing in the breeze."
        favorite_poem_input = st.text_area("Enter your favorite poem", placeholder=favorite_poem_example, height=160)

        if st.button('Get Recommendation'):
            if uploaded_file is None and not favorite_poem_input:
                st.error('Please upload a text file or enter your favorite poem')
            else:
                with st.spinner('Getting recommendations...'):
                    if uploaded_file is not None:
                        # Read the uploaded file
                        favorite_poem = uploaded_file.read().decode("utf-8")
                    else:
                        favorite_poem = favorite_poem_input

                    # Code for recommendation system goes here
                    recommendations = get_recommendations(favorite_poem)

                    # Display the recommendations
                    st.markdown('___')
                    st.write('Recommended Poems:')
                    for index, row in recommendations.iterrows():
                        with st.expander(f"{row['Title']} by {row['Author']}"):
                            poem_content = read_poem(row['Poem'])
                            tags = ast.literal_eval(row['Tags'])
                            st.markdown(f"**Title:** {row['Title']}<br>**Author:** {row['Author']}", unsafe_allow_html=True)
                            st.markdown(poem_content, unsafe_allow_html=True)
                            st.markdown(f"**tags:** {', '.join(tags)}", unsafe_allow_html=True)

