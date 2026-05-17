from crewai import Crew, Agent, Task, LLM
import litserve as ls
from local_llm import LocalOllamaLLM
from fire_crawl import fc

from fire_crawl import FireCrawl
from vector_search import VectorDBSearchTool

class AgenticRAGAPI(ls.LitAPI):
    def setup(self, device):
        llm = LocalOllamaLLM()

        researcher_agent = Agent(
            role = "Senior Researcher",
            goal = "Research about the user query",
            backstory = "You are a skilled researcher",
            tools = [FireCrawl, VectorDBSearchTool],
            llm = llm
            )
        
        researcher_task = Task(description= "Research about: {query}", agent= researcher_agent)