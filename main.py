import torch
from transformers import pipeline
import tensorflow_hub as hub
import pandas as pd
import joblib

def embed(texts):
    return model(texts)

def recommend(text: str):
    text.replace("\n", " ")
    emb = embed([text])
    neighbors = nn.kneighbors(emb, return_distance=False)[0]
    return df.iloc[neighbors]

def init_pipeline():
    return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

def generate_poem(messages, do_sample, temperature, top_k, top_p):
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    outputs = pipe(prompt, max_new_tokens=100, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p)
    return outputs[0]["generated_text"].split("<|assistant|>\n")[1]


if __name__ == '__main__':    
    option = str(input("'recommend poem' or 'generate poem':\n"))
    
    if option == "recommend poem":
        input= str(input("give a title part of poem anythings\n"))
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") # load the universal sentence encoder
        df = pd.read_pickle("poems_recommendation\data_for_recommandations.pkl") # load poems for recommandations
        nn = joblib.load('poems_recommendation\\5nn_model.joblib') # load the knn model trained on poems
        recommendations_df = recommend(input) ## to be displayde in Streamlit app
        print(recommendations_df) # print the recommendations (for now)
        
    elif option == "generate poem":
        poem_title = str(input("give poem title\n"))
        pipe = init_pipeline()
        messages = [
                    {
                        "role": "system",
                        "content": "You are a poet who writes poetry based on the title",
                    },
                    {"role": "user", "content": f"write a mini poetry titled \"{poem_title}\""},
                    ]
        generated_poem = generate_poem(messages, do_sample=True, temperature = 0.7, top_k=50, top_p=0.95)
        print(generated_poem)

