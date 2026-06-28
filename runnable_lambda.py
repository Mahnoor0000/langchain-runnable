# converts a normal Python function into a runnable component.

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def word_counter(text: str) -> int:
    return len(text.split())


model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY")
)



runnable_word_counter = RunnableLambda(word_counter)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

joke_generator_chain = RunnableSequence(prompt, model, parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_counter)
}
)

final_chain = RunnableSequence(
    joke_generator_chain,
    parallel_chain
)

result = final_chain.invoke({"topic": "AI"})
print(result)