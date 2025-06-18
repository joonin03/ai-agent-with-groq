from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.tools import DuckDuckGoSearchRun

# 🔑 .env 파일로부터 GROQ_API_KEY 로드 (또는 직접 문자열로 삽입)
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")  

# ✅ Groq 기반 LLM 설정
llm = ChatOpenAI(
    openai_api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
    model="gemma2-9b-it",  
    temperature=0.7
)

# ✅ DuckDuckGo 웹 검색 도구 설정
search_tool = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search_tool.run,
        description="실시간 정보가 필요할 때 사용합니다. 예: 최신 뉴스, 대학 정보 등"
    )
]

# ✅ LangChain Agent 생성
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ✅ 실행
response = agent.invoke("동국대학교 첨단 융합대학에 대해서 알려줘")
print("\n🤖 결과:\n", response)
