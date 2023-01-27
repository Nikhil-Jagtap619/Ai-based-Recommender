import streamlit as st
import pandas as pd
import pickle
import time

hide_streamlit_style = """

            <style>
            footer {visibility: hidden;}
            body {color: #F2F2F2}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

s = f"""
    <style>
    div.stButton > button:first-child {{ border: 5px solid {"#0077FD"}; border-radius:20px 20px 20px 20px; }}
    <style>
    """
st.markdown(s, unsafe_allow_html=True)

from PIL import Image
# st.set_page_config(
#     page_title="NABL App",
#     page_icon="mail")
image = Image.open('icon.png')
st.image(image, width=200)


#import models
tfidf = pickle.load(open("tf_vectorizer_char.pkl", "rb"))
model_NBC = pickle.load(open("NBC.pkl","rb"))
data = pd.read_excel("level_3.xlsx")
# data = data.sample(20)
# data2 = data.reset_index()
# print(data2)


def preprocess(txt):
    x = txt.replace(" ","")
    vector = tfidf.transform([x])
    # print(vector)
    return vector



def Status(user):
    x = user
    x = preprocess(x)
    x = model_NBC.predict(x)
    # y = model_NBC.predict_proba(y)
    # print(y[0])
    return x[0]

def reco(data):
    arr = []
    for i in data["MUunit"]:
        re = Status(i)
        arr.append(re)
        
    # arr = pd.DataFrame(arr, columns=["Recommended Unit"]).T
    return arr

def color_survived(val):
    # color = '#0077b6' if val else '#bc4749'
    color = '#6a994e' if val else '#bc4749'
    return f'background-color: {color}'


def main():
    st.title("Demo: AI-Based Recommendation")
    st.markdown("<h3>Cognitive Automation, Focused Only On <span style=\"color: #0077FD\">Micro Units</span></h3>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        user = st.text_input("Enter Your Output Value:")
    if st.button("submit"):
      result = Status(user)
      with st.spinner('Wait for it...'):
        time.sleep(0.2)
        # st.write(f" Corrected Value: {result}") # make it bold
        col3, col4 = st.columns(2)
        with col3:
            st.text_input(label="Recommended Value:", value=result)
    #   prob_df = pd.DataFrame(prob, index=["probability"], columns=model_NBC.classes_)
    #   st.write(prob_df)
    st.markdown("<hr/>", unsafe_allow_html=True)
    arr = reco(data=data)
    df = data
    st.markdown("The AI recognises mistyped Units and recommends the appropriate standards.")
    if st.checkbox("Recommendation"):
        df["Recommended Unit"] = arr
        st.write(df.style.applymap(color_survived, subset="Recommended Unit"))
    else:
        st.write(df)
    st.markdown("<br/>", unsafe_allow_html=True)
    st.write("## Thank you for Visiting \nProject by Nikhil J")
    st.markdown("<h1 style='text-align: right; color: #d7e3fc; font-size: small;'><a href='https://github.com/Nikhil-Jagtap619/'>Looking for Source Code?</a></h1>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()