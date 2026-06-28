from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableParallel, RunnableSequence
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm1 = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY")
)

llm2 = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY")
)
prompt1 = PromptTemplate(
    template = 'write a tweet about {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'write a LinkedIn post about {topic}',
    input_variables = ['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'tweet': RunnableSequence(prompt1, llm1, parser),
        'linkedin': RunnableSequence(prompt2, llm2, parser)
    }
)

parallel_result = parallel_chain.invoke({"topic": "AI in healthcare"})
print("Generated Tweet:", parallel_result['tweet'])
print("Generated LinkedIn Post:", parallel_result['linkedin'])