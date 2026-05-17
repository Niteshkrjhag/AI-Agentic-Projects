from dotenv import load_dotenv
import os
from firecrawl import Firecrawl

load_dotenv()

class FireCrawl:
    def __init__(self):
        self.api_key = os.getenv("FIRECRAWL_API_KEY")
        self.firecrawl = Firecrawl(api_key = self.api_key)

    def __call__(self, query: str):
        return self.firecrawl.search(query)



if __name__=="__main__":
    fc = FireCrawl()
    print(fc("AI news"))

