# runnable that works like an if-else condition inside a chain.

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableBranch, RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser



model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY")
)

prompt1 = PromptTemplate(
    template='write a detailed report about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='write a short summary of this text  {text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_generator_chain = RunnableSequence(prompt1, model, parser)
branch_chain = RunnableBranch(
   (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
   RunnablePassthrough()
)

final_chain = RunnableSequence(
    report_generator_chain,
    branch_chain
)
final_result = final_chain.invoke({"topic": "breaking bad"})

print("Final Result:", final_result)