from crewai import Crew, Agent, Task, LLM
import litserve as ls

from fire_crawl import fc
import vector_search

class AgenticRAGAPI(ls.LitAPI):
    def setup(self, device):
        llm = LLM(model="ollama/qwen3")