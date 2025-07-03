from crewai import Agent, Task, Crew, LLM 
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import tool
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="CrewAI - Content Generation", page_icon="🧠", layout="wide")

st.title("Content Generation with CrewAI")
st.markdown("Welcome to the CrewAI demo! This app demonstrates how CrewAI agents can collaborate to generate content efficiently.")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    # Use text_input instead of text_area if multiline is not needed
    topic = st.text_input("Topic", "Generative AI",
                          placeholder="Enter the topic for content generation")

    st.markdown(f"### LLM Configuration")
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)

    generate_button = st.button("Generate Content", type="primary")

# Information about the demo
with st.expander("Show Details"):
    st.markdown("""
        This demo showcases how CrewAI can be used to generate content collaboratively using AI agents.
        - **Senior Research Analyst:** Conducts research on the given topic.
        - **Content Writer:** Writes an article based on the research findings.
        - Tools include AI language models and web search capabilities.
    """)

# Function to generate content using CrewAI
def generate_content(topic):
    # Initialize AI model
    llm = LLM(model="groq/gemma2-9b-it")
    duckduckgo = DuckDuckGoSearchRun()


    # Tool 2
    # search_tool = DuckDuckGoSearchRun()
    @tool("duckduckgo_search_tool")
    def duckduckgo_search(query: str) -> str:
        """Search the web using DuckDuckGo and return relevant links and summaries."""
        return duckduckgo.run(query)

    # Define AI agents
    senior_research_analyst = Agent(
        role="Senior Research Analyst",
        goal=f"Conduct in-depth research on {topic}",
        backstory="You are a highly experienced AI and industry analyst, known for thorough research.",
        allow_delegation=False,
        tools=[duckduckgo_search],
        llm=llm
    )

    # ✍️ Agent 2: Content Writer
    content_writer = Agent(
        role="Content Writer",
        goal=f"Write a compelling and informative article on {topic}",
        backstory="You are a talented writer who transforms research into engaging content.",
        allow_delegation=False,
        llm=llm
    )

    # 🧪 Task 1: Research
    research_task = Task(
        description=f"Research all key trends, technologies, and applications related to {topic}.",
        expected_output=f"A well-organized summary of the current state of {topic} with relevant examples and references.",
        agent=senior_research_analyst
    )

    # 📝 Task 2: Write
    writing_task = Task(
        description=f"Write an insightful, well-structured article based on the research about {topic}.",
        expected_output=f"A detailed, engaging article with an introduction, body, and conclusion on {topic}.",
        agent=content_writer
    )

    # 🧩 Assemble the crew
    crew = Crew(
        agents=[senior_research_analyst, content_writer],
        tasks=[research_task, writing_task]
    )


    # Execute tasks and return result
    return crew.kickoff(inputs={"topic": topic})

# Trigger content generation on button click
if generate_button:
    with st.spinner("Generating content..."):
        try:
            result = generate_content(topic)
            st.markdown(f"### Content Generation Result")
            st.markdown(result)

            # Download option for the generated content
            st.download_button(
                label="Download Content",
                data=result.raw,
                file_name=f"{topic.replace(' ', '_')}_article.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer section
st.markdown("---")
st.markdown("Built with ❤️ by [Thatipalli Sai Shiva](https://saishiva-portfolio.netlify.app/)")
