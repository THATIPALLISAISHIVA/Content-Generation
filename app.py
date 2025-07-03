from crewai import Agent, Task, Crew, LLM 
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import tool


from dotenv import load_dotenv

load_dotenv()

# Initialize the CrewAI Agent

topic = "Medical Industry using Generative AI"

# Basic configuration - Tool 1
llm = LLM(
    model="groq/gemma2-9b-it"
)

duckduckgo = DuckDuckGoSearchRun()


# Tool 2
# search_tool = DuckDuckGoSearchRun()
@tool("duckduckgo_search_tool")
def duckduckgo_search(query: str) -> str:
    """Searches the web using DuckDuckGo and returns top results."""
    return duckduckgo.run(query)

# Agent 1
senior_research_analyst = Agent(
    role="Senior Research Analyst",
    goal = "Research on the Medical Industry using Generative AI",
    backstory = "You are Senior Research Analyst with 5 years of experience in the field of AI and Machine Learning. You have worked on multiple projects related to the medical industry and have a good understanding of the domain.",
    allow_delegation = False,
    verbose = True,
    tools = [duckduckgo_search],
    llm = llm
)

# Agent 2

content_writer = Agent(
    role="Content Writer",
    goal = "Write a detailed article on the Medical Industry using Generative AI",
    backstory = "You are a Content Writer with 3 years of experience in writing articles on various topics. You have written articles on the medical industry in the past and have a good understanding of the domain.",
    allow_delegation = False,
    verbose = True,
    llm = llm
)

# Research Task

research_task = Task(
    description = "Research on the Medical Industry using Generative AI",
    expected_output = "A detailed summary of the current trends and applications of Generative AI in the Medical Industry",
    agent = senior_research_analyst
)

# Content Generation Task

writing_task = Task(
    description = "Write a detailed article on the Medical Industry using Generative AI",
    expected_output = "A well-researched article on the current trends and applications of Generative AI in the Medical Industry",
    agent = content_writer
)

# Crew

crew = Crew(
    agents = [senior_research_analyst, content_writer],
    tasks = [research_task, writing_task],
    verbose = True
)

result = crew.kickoff(inputs={"topic": topic})

print(result)
