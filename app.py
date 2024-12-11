from flask import Flask, render_template, request, jsonify
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initializing Flask app
app = Flask(__name__)

# LLM and Retrieval Setup
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
loader = WebBaseLoader("https://dph.illinois.gov/topics-services/diseases-and-conditions/diseases-a-z-list.html")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs[:50])
vectors = FAISS.from_documents(final_documents, embeddings)

retriever = vectors.as_retriever()

# Creating the chain for retrieval and answering
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# Route for the chatbot interface
@app.route("/")
def index():
    return render_template("index.html")  


# Route to handle AJAX POST request
@app.route("/get", methods=["POST"])
def chatbot_response():
    user_message = request.form["msg"]

    if not user_message:
        return jsonify("Error: No input provided"), 400

    # Geting response from the retrieval chain
    response = retrieval_chain.invoke({"input": user_message})
    answer = response["answer"]
    return jsonify(answer)


if __name__ == "__main__":
    app.run(debug=True)
