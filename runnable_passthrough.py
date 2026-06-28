# SIMPLY RETURN INPUT AS OUTPUT WITH NO MODIFICATIONS


from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY")
)

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

joke_generator_chain = RunnableSequence(prompt1, llm, parser)

parallel_chain = RunnableParallel(
    {
        'joke' : RunnablePassthrough(),
        'explanation' : RunnableSequence(prompt2, llm, parser)
    }
)

final_chain = RunnableSequence(
    joke_generator_chain,
    parallel_chain
)

result = final_chain.invoke({"topic": "programming"})
print(result)