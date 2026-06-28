from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate


load_dotenv()

# 1. Load PDF
loader = PyPDFLoader("Sample.pdf")
documents = loader.load()

# 2. Split PDF text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store chunks in FAISS vector database
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# 6. Load Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# 7. Prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful PDF assistant.

Answer the question using only the given PDF context.
If the answer is not found in the context, say:
"I could not find this information in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""
)

# 8. Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever)

print("PDF Reader is ready!")
print("Type 'exit' to stop.\n")

while True:
    question = input("Ask question: ")

    if question.lower() == "exit":
        break

    response = qa_chain.invoke({
        "query": question
    })

    print("\nAnswer:")
    print(response["result"])

    

    print("-" * 50)