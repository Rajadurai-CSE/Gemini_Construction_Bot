from image_gen import generate_image_fun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from configure import configure
configure()

def summarize(docs,filename):
    
    prompt_template = """Write a concise summary of the following:" {text}" CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")
    llm_chain = LLMChain(llm=llm,prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    response = stuff_chain.invoke(docs)
    modified_output = f'Summary of the Document {filename} \n\n' + response['output_text']
    return modified_output

def image_prompt_generator(content):
    content = f"From the summary of the document generate image prompt that can be useful to visualize important points \n content : {content}"
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(content)
    output = generate_image_fun(response.text)
    print(response.text)
    return output
