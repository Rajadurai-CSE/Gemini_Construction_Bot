#Check Search Results
# Delete Documents function
# Summary and Image after pdf processed
import markdownify
import streamlit as st
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
import replicate
from langchain.prompts import PromptTemplate
# import pymupdf   # PyMuPDF
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\rajad\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
from PIL import Image
from streamlit_mic_recorder import  speech_to_text
import io
import shutil
from llama_parse import LlamaParse
from langchain.chains import LLMChain
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


parser = LlamaParse(
      # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
)

def intialize_session_state():
  if "generated" not in st.session_state:
    st.session_state['generated'] = []
  if 'past' not in st.session_state:
    st.session_state['past'] = []
  if 'generated_images' not in st.session_state:
    st.session_state['generated_images'] = {}
  if 'chain' not in st.session_state:
     st.session_state['chain'] = get_conversational_chain()
  if 'user_input_images' not in st.session_state:
      st.session_state['user_input_images'] = {}
      # Initialize session state for vector_store and uploaded_files if not already done
  if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
  if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
  if 'current_image_input' not in st.session_state:
       st.session_state.current_image_input = []
  if 'summary_docs' not in st.session_state:
       st.session_state.summary_docs = []
  if 'img_fsum' not in st.session_state:
       st.session_state.img_fsum = []
    
    #have it as list and do the necessary operation



# def get_pdf_text(pdf_docs):

#     doc = pymupdf.open(stream=pdf_docs)
#     full_text = ""
    
#     for page in doc:
#         text = page.get_text()
#         if text:
#             full_text += text
#         else:
#             for img in page.get_images(full=True):
#                 xref = img[0]
#                 base_image = doc.extract_image(xref)
#                 image_bytes = base_image["image"]
#                 image = Image.open(io.BytesIO(image_bytes))
#                 ocr_text = pytesseract.image_to_string(image)
#                 full_text += ocr_text
                

#     return full_text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to summarize text using an LLM
def summarize(docs,filename):
    
    prompt_template = """Write a concise summary of the following:" {text}" CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")
    llm_chain = LLMChain(llm=llm,prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    response = stuff_chain.invoke(docs)
    modified_output = f'Summary of the Document {filename} \n\n' + response['output_text']
    st.session_state['summary_docs'].append(modified_output) 
    return f"Summary of Documents \n"+ response['output_text']

def image_prompt_generator(content):
    content = f"From the summary of the document generate image prompt that can be useful to visualize important points \n content : {content}"
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(content)
   
    output = generate_image_fun(response.text)
    st.session_state['img_fsum'].append(output)
    return


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store




def get_conversational_chain():

    # model = genai.GenerativeModel(model='gemini-1.5-flash')

   

    model = ChatGoogleGenerativeAI(model ='gemini-1.5-flash',temperature=0.5,convert_system_message_to_human=True)
    # model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ('system',' You are a construction expert. Your work is to assist contractors, architects , general people who are interested in buying home . Your task will be from providing insights on budgets of new projects, aiding architects in designing to guiding people to buy new home . Provide the audience with relavant information from the given context. You will also be given images which you can analyze and answer. If you feel you are not able to fully understand the context or image, then just say i am not able to answer and provide some alternatives. Be precise, clear to the audience. If there is not context given or image given, then try to answer the question with your general knowledge related to construction Industry, be a friendly answering casual questions but not to questions that are unappropriate to construction industry'),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human','\n{input}')
            
            ]
    )
    
    memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=3
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

#Image Generator Function to Generate Image




def generate_image_fun(prompt):
    input = {
    "width": 768,
    "height": 768,
    "prompt": prompt,
    "refine": "expert_ensemble_refiner",
    "apply_watermark": False,
    "num_inference_steps": 25
    }
    output = replicate.run(
    "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
    input=input
    )
    return output
    

def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings)
    return db



def update_state(text):
    st.session_state['input'] = text

def display_chat_history():

  reply_container = st.container()
  container = st.container()

  vector_store = st.session_state['vector_store']
  chain = st.session_state['chain']
  current_images = st.session_state['current_image_input']
  if current_images:
      current_image_dict = {
      'type':'PIL image',
      'content':current_images
      }
  else:
      current_image_dict = None

      
#   images = st.session_state['images']

  with container:
    text = speech_to_text(start_prompt='Start Recording',stop_prompt='Stop Recording',language='en', use_container_width=True, just_once=True, key='speech')

    with st.form(key="my_form",clear_on_submit=True):
      user_input = st.text_input("Question:",placeholder="Ask",key='input')
      col1,col2 = st.columns(2)
      with col1:
          submit_button = st.form_submit_button(label="Send")
      with col2:
          generate_image = st.form_submit_button(label='Generate Image')

    if generate_image and user_input!='':
        with st.spinner("Processing query..."):
            st.session_state['past'].append(user_input) #Add Image to the prompt
            output = generate_image_fun(prompt=user_input)
            st.session_state['generated_images'][len(st.session_state['past'])-1] = output
            st.session_state['generated'].append("There you go!!")


    if (submit_button and user_input) or text:
      with st.spinner("Processing query..."):
        if text:
            user_input = text
        if vector_store:
            new_db = load_vectorstore()
            context = new_db.similarity_search(user_input,k=1) 
        else:
            context = ''
        

        combined_input = f"Context:\n{context}\nQuestion:\n{user_input}\nImages:\n{current_image_dict}" #Add --> User Inputted Images
        output =  chain.invoke({'input':combined_input})

        # output = conversation_chat(user_input,context,chain)
        st.session_state['past'].append(user_input) #Add Image to the prompt
        st.session_state['generated'].append(output['text'])
        st.session_state['current_image_input'] = []

    
    with reply_container:

        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i],is_user=True  ,key=str(i)+ "_user",avatar_style="thumbs")
            if i in st.session_state['user_input_images'].keys():
                images = st.session_state['user_input_images'][i]
                for i in range(len(images)):
                    st.image(images[i],caption = 'User Input')
            message(st.session_state['generated'][i],key = str(i), avatar_style="fun-emoji")
            if i in st.session_state['generated_images'].keys():
                st.image(st.session_state['generated_images'][i], caption="Generated Image")
        


def main():
    intialize_session_state()
    st.set_page_config(page_title="Construction Expert", layout="wide")
    st.header("Construction Expert")
    display_chat_history()
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if pdf_docs:
            if not os.path.exists('files'):
                os.makedirs('files')
            for pdf in pdf_docs:
                if pdf.name not in st.session_state.uploaded_files:
                    with open(os.path.join("files",pdf.name),"wb") as f:
                        f.write(pdf.getbuffer())
                    file_path = os.path.join('files', pdf.name)
                    st.session_state.uploaded_files[pdf.name] = file_path
        content = []
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    for file_name, file_path in st.session_state.uploaded_files.items():
                        # doc = PyPDFLoader(file_path)
                        # loader = doc.load()
                        docs = parser.load_data(file_path)
                        # print(docs) #Metadata --> Source and Markdown to text
                        # markdown_string = markdownify.markdownify(docs[0].text)
                        # print(markdown_string)
                        text_chunks = get_text_chunks(docs[0].text)
                        content.extend(text_chunks)
                        doc_content = [Document(page_content=t) for t in text_chunks]
                        summarized = summarize(doc_content,file_name)
                        print(summarized)
                        image_prompt_generator(summarized)
            index = get_vector_store(content)
            if index:
                st.session_state.vector_store = True
                st.success("Done")
                reply_container = st.container()
                with reply_container:
                    for k in range(len(st.session_state['img_fsum'])):
                        message(st.session_state['summary_docs'][k],is_user=True  ,key=str(k)+ "_sum",avatar_style="thumbs")
                        st.image(st.session_state['img_fsum'][k], caption="Generated Image")
            else:
                st.error("Failed to create vector store")
           

    
        

        
         # Display uploaded files with delete option
        # st.subheader("Uploaded Files:")
        # for file_name in st.session_state.uploaded_files.keys():
        #     col1, col2 = st.columns([4, 1])
        #     col1.write(file_name)
        #     if col2.button("Delete", key=file_name):
        #         os.remove(st.session_state.uploaded_files[file_name])
        #         del st.session_state.uploaded_files[file_name]
        #         # delete_vector_store()  # Delete vector store when a file is deleted
                # st.experimental_rerun()

# Delete Function to delete the faiss index associated with the user document if the user deletes a document
# Delete Files from files directory after session over
#      Solution --> When the app starts delete all the files
#                   Session Persist --> When reloading do not delete the files
#      Add Delete option to each file and remove from directory



if __name__ == "__main__":
    main()
