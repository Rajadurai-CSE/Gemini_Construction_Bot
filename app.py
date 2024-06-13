import streamlit as st
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pandas as pd
from PIL import Image
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import pymupdf   # PyMuPDF
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\rajad\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
from PIL import Image
from streamlit_mic_recorder import  speech_to_text
import io
from langchain.chains import LLMChain
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def intialize_session_state():
  if "generated" not in st.session_state:
    st.session_state['generated'] = []
  if 'past' not in st.session_state:
    st.session_state['past'] = []
  if 'chain' not in st.session_state:
     st.session_state['chain'] = get_conversational_chain()



def get_pdf_text(pdf_docs):

    doc = pymupdf.open(stream=pdf_docs)
    full_text = ""
    
    for page in doc:
        text = page.get_text()
        if text:
            full_text += text
        else:
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(image)
                full_text += ocr_text

    return full_text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain():

   

    model = ChatGoogleGenerativeAI(model ='gemini-1.5-flash',temperature=0.5,convert_system_message_to_human=True,max_output_tokens=150)
    # model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ('system',' You are a construction expert. Your work is to assist contractors, architects , general people who are interested in buying home . Your task will be from providing insights on budgets of new projects, aiding architects in designing to guiding people to buy new home . Provide the audience with relavant information from the given context. If you feel you are not able to fully understand the context, then just say i am not able to answer and provide some alternatives. Be precise, clear to the audience.If there is not context given, then try to answer the question with your general knowledge related to construction Industry, be a friendly answering casual questions but not to questions that are unappropriate to construction industry'),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human','\n{input}')
            
            ]
    )
    
    memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=10
    )

    chain = LLMChain(
    llm=model,
    prompt=prompt_template,
    memory=memory,
    verbose=True
    )
    # prompt = prompt_template.format(context=context, question=question)
    # chat = model.generate_content(prompt)
    # # chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    # conversation_with_summary = ConversationChain(
    # llm=model,
    # prompt=prompt,
    # memory=memory,
    # verbose=True
    # )
    return chain



def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings)
    return db


def store_to_df(store):
    v_dict = store.docstore._dict
    print(v_dict)
    data_rows = []
    for k in v_dict.keys():
        doc_name = v_dict[k]
        print(doc_name)

def delete_documents(store,document_name):
    vector_df = store_to_df(store)
    chunk_list = vector_df.loc[vector_df['document_name']==document_name]['chunk_id']
    store.delete(chunk_list)

def update_state(text):
    st.session_state['input'] = text

def display_chat_history():

  reply_container = st.container()
  container = st.container()

  vector_store = st.session_state['vector_store']
  chain = st.session_state['chain']


  with container:
    text = speech_to_text(start_prompt='Start Recording',stop_prompt='Stop Recording',language='en', use_container_width=True, just_once=True, key='STT')


    with st.form(key="my_form",clear_on_submit=True):
      user_input = st.text_input("Question:",placeholder="Ask",key='input')
      submit_button = st.form_submit_button(label="Send")
   

    if (submit_button and user_input) or text:
      with st.spinner("Processing query..."):
        if text:
            user_input = text
        if vector_store:
            new_db = load_vectorstore()
            context = new_db.similarity_search(user_input) 
        else:
            context = ''
        
        combined_input = f"Context:\n{context}\nQuestion:\n{user_input}"
        output =  chain.invoke({'input':combined_input})

        # output = conversation_chat(user_input,context,chain)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output['text'])
     
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i],is_user=True  ,key=str(i)+ "_user",avatar_style="thumbs")
                message(st.session_state['generated'][i],key = str(i), avatar_style="fun-emoji")


def main():
    intialize_session_state()
  
    st.set_page_config(page_title="Construction Expert", layout="wide")
    st.header("Construction Expert")

    # Initialize session state for vector_store and uploaded_files if not already done
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    #have it as list and do the necessary operation

    display_chat_history()


    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if pdf_docs:
            for pdf in pdf_docs:
                if pdf.name not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files[pdf.name] = pdf

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = ''
                    for pdf in pdf_docs:
                        pdf_binary = pdf.read()
                        raw_text += get_pdf_text(pdf_binary)
                    text_chunks = get_text_chunks(raw_text)
                    index = get_vector_store(text_chunks)
                    if index:
                        st.session_state.vector_store = True
                        st.success("Done")
                    else:
                        st.error("Failed to create vector store")

         



if __name__ == "__main__":
    main()
