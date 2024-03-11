from langchain_community.tools.google_finance.tool import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper

from dotenv import load_dotenv

load_dotenv("../.env")

if __name__ == '__main__':
    result = ''
    # result = BrowserTools.scrape_and_summarize_website("https://python.langchain.com/docs/modules/model_io/quick_start")
    # result = CalculatorTools.calculate("200*7")
    # result = SearchTools.search_internet("OpenAI")
    # result = SECTools.search_10q("O-P-E-N|What were the total revenues for the last quarter")
    # result = YahooFinanceNewsTool()._run("Apple")
    result = GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper())._run("Apple")
    print(result)
