# 🧠 CrewAI - Content Generation App

Welcome to the **CrewAI Content Generation App**, a Streamlit-based demo that showcases how multiple AI agents can collaborate to research and write high-quality content on any given topic using [CrewAI](https://github.com/joaomdmoura/crewAI).

This app combines **LLMs**, **web search tools**, and a multi-agent framework to produce insightful articles with minimal input.

---

## 🚀 Features

- 🔍 **Senior Research Analyst Agent**  
  Conducts in-depth research using DuckDuckGo on the given topic.

- ✍️ **Content Writer Agent**  
  Writes an engaging and informative article based on research findings.

- 🌐 **DuckDuckGo Search Tool**  
  Used to retrieve the latest information and references from the web.

- ⚙️ **Custom LLM Configuration**  
  Easily adjust temperature and model behavior from the Streamlit sidebar.

- 📄 **Markdown Output**  
  Generated content is displayed in markdown and available for download.

---

## 🧰 Tech Stack

- **Python**
- **Streamlit**
- **CrewAI**
- **LangChain**
- **DuckDuckGoSearchRun** (for web search)
- **LLM** (e.g., `groq/gemma2-9b-it`)
- **dotenv** (for managing environment variables)

---

## 🖥️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/crewai-content-generator.git
   cd crewai-content-generator
