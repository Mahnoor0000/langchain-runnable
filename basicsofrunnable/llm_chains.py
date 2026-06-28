from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate


load_dotenv()

# Load the LLM using Groq
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)

# Create a Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}."
)

# Create an LLMChain
chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# Run the chain
topic = input("Enter a topic: ")

output = chain.run(topic)

print("Generated Blog Title:", output)