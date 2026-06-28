from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

llm = ChatGroq(
    model="gpt-4o-mini",
    api_key=os.environ.get("GROQ_API_KEY")
)

topic = input("Enter a topic for the blog title: ")

prompt = ChatPromptTemplate(
    template = 'write blog title about {topic}',
    input_variables = ['topic']
)



format_prompt = prompt.format_prompt(topic=topic)
response = llm.invoke(format_prompt.to_messages())

print("Generated Blog Title:", response.content)