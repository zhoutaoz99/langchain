from crewai import Agent
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper

from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools
from tools.sec_tools import SECTools
from chatglm.chat_zhipu_ai import ChatZhipuAI

class StockAnalysisAgents():
  def financial_analyst(self):
    return Agent(
      role='最佳财务分析师',
      goal="""用您的财务数据和市场趋势分析让所有客户印象深刻""",
      backstory="""这是最有经验的财务分析师，拥有丰富的股票市场分析和投资策略专业知识，目前正为一个非常重要的客户工作。""",
      verbose=True,
      tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        CalculatorTools.calculate,
        SECTools.search_10q,
        SECTools.search_10k
      ],
      llm=ChatZhipuAI()
    )

  def research_analyst(self):
    return Agent(
      role='员工研究分析师',
      goal="""在收集、解读数据方面做到最好，并以此令您的客户惊叹""",
      backstory="""作为公认的最佳研究分析师，您擅长筛选新闻、公司公告和市场情绪。现在您正在为一个非常重要的客户工作。""",
      verbose=True,
      tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        SearchTools.search_news,
        GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper()),
        SECTools.search_10q,
        SECTools.search_10k
      ],
      llm=ChatZhipuAI()
  )

  def investment_advisor(self):
    return Agent(
      role='私人投资顾问',
      goal="""通过全面的股票分析和完整的投资建议给您的客户留下深刻印象""",
      backstory="""您是最有经验的投资顾问，能够结合各种分析洞见来制定战略投资建议。现在您正在为一个您需要给其留下深刻印象的非常重要的客户工作。""",
      verbose=True,
      tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        SearchTools.search_news,
        CalculatorTools.calculate,
        GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper())
      ],
      llm=ChatZhipuAI()
    )