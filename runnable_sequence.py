from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY")
)

prompt = PromptTemplate(
    template = 'write a joke about {topic}',
    input_variables = ['topic']
)

parser = StrOutputParser()


chain = RunnableSequence(
    prompt, llm, parser
)

result = chain.invoke({"topic": "programming"})

print("Generated Joke:", result)