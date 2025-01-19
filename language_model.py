import os
import pickle
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationChain

# Setup environment
load_dotenv(find_dotenv())

# Setup and return conversation chain
def load_conversation_chain():
    # Use a custom model (ChatGroq in your case)
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    
    # Define memory and use the correct memory_key for conversation
    memory = ConversationBufferMemory()
    
    # Create the conversation chain using memory, prompt, and llm
    conversation = ConversationChain(llm=llm, memory=memory)
    return conversation
