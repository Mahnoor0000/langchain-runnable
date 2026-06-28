from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

pdf_path = "Sample.pdf"

print("Loading PDF...")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("Splitting PDF...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful PDF assistant.

Answer the question using only the PDF context.

If the answer is not in the PDF, say:
"I could not find this information in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""
)

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("PDF reader is ready!")
print("Type exit to stop.\n")

while True:
    question = input("Ask a question: ")

    if question.lower() == "exit":
        break

    answer = chain.invoke(question)

    print("\nAnswer:")
    print(answer)
    print()