# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import CharacterTextSplitter
# from llama_parse import LlamaParse 

# load_dotenv()

# parser = LlamaParse(
#       # can also be set in your env as LLAMA_CLOUD_API_KEY
#     result_type="markdown",  # "markdown" and "text" are available
#     verbose=True,
# )

# loader = DirectoryLoader("D:/Construction_Bot/files",loader_cls = parser)
# pages = loader.load_and_split()
# print(pages)
# print("-----------------")
# text_splitter = CharacterTextSplitter(chunk_size  = 10, chunk_overlap = 0)
# docs = text_splitter.split_documents(pages)
# print(docs)


from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
load_dotenv()
reader = SimpleDirectoryReader(input_dir="D:/Construction_Bot/files")
documents = reader.load_data()