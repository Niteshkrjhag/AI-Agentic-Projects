from dotenv import load_dotenv
import os
from firecrawl import Firecrawl

load_dotenv()
api_key = os.getenv("FIRECRAWL_API_KEY")

fc = Firecrawl(api_key=api_key)

result = fc.search("AI news")

print(result)