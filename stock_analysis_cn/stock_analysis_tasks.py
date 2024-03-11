from crewai import Task
from textwrap import dedent


class StockAnalysisTasks():
    def research(self, agent, company):
        return Task(description=dedent(f"""
        收集并总结与该股票及其行业相关的最新新闻文章、新闻稿和市场分析。 特别关注任何重大事件、市场情绪和分析师的看法。同时包括即将到来的事件，如收益报告等。

        您的最终答案必须是一份报告，包括最新新闻的全面总结、市场情绪的任何显著变化，以及对股票可能产生的影响。
        确保返回股票的股票代码。

        {self.__tip_section()}

        确保使用尽可能最新的数据。

        客户选择的公司： {company}
      """),
                    agent=agent
                    )

    def financial_analysis(self, agent):
        return Task(description=dedent(f"""
        对股票的财务健康状况和市场表现进行彻底分析。 这包括检查关键的财务指标，如市盈率（P/E）、每股收益（EPS）增长、收入趋势和负债权益比率。 同时，分析该股票与其行业同行以及整体市场趋势相比的表现。

        您的最终报告必须扩展所提供的摘要，但现在包括对股票财务状况的清晰评估，其优势与劣势，以及它在当前市场情况下的竞争对手表现如何。{self.__tip_section()}

        请确保使用可能获取的最新数据。
      """),
                    agent=agent
                    )

    def filings_analysis(self, agent):
        return Task(description=dedent(f"""
        分析EDGAR数据库中该股票最新的10-Q和10-K文件。 重点关注管理层讨论与分析、财务报表、内幕交易活动以及任何披露的风险等关键部分。 提取可能影响股票未来表现的相关数据和洞见。

        您的最终答案必须是一个扩展报告，现在还要突出这些文件中的重大发现，包括对您的客户而言的任何红旗或积极指标。
        {self.__tip_section()}        
      """),
                    agent=agent
                    )

    def recommend(self, agent):
        return Task(description=dedent(f"""
        审查并综合财务分析师和研究分析师提供的数据分析。 将这些洞见结合起来，形成一个全面的投资建议。您必须考虑所有方面，包括财务健康状况、市场情绪以及来自EDGAR文件的质量数据。

        确保包括展示内幕交易活动和即将到来的事件（如收益报告）的部分。

        您的最终答案必须是针对您的客户的一个建议。它应该是一个非常详细的完整报告，提供一个明确的投资立场和策略，并附有支持证据。 确保报告美观且格式良好，以便您的客户阅读。
        {self.__tip_section()}
      """),
                    agent=agent
                    )

    def __tip_section(self):
        return "如果您做到最好，我会给您1万美元的佣金！"