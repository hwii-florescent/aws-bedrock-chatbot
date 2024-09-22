from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st

os.environ["AWS_PROFILE"] = "hwiiyy"

# bedrock client
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1' 
)

modelID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

llm = ChatBedrock(
    model_id=modelID,
    client=bedrock_client
)

def my_chatbot(language, freeform_text):
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template="You are a chatbot. You are in {language}.\n\n{freeform_text}"
    )

    # New RunnableSequence approach
    bedrock_chain = prompt | llm

    response = bedrock_chain.invoke({
        'language': language, 
        'freeform_text': freeform_text
    })
    return response

st.title("Chatbot AWS Bedrock")

language = st.sidebar.selectbox("Language", ["English", "French", "German", "Italian", "Spanish"])

if language:
    freeform_text = st.sidebar.text_area(label="Type in your question", max_chars=200)

if freeform_text:
    response = my_chatbot(language, freeform_text)
    st.write(response.content)
