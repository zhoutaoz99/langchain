from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic.v1 import Extra

from zhipuai import ZhipuAI


class ZhipuAILLM(LLM):
    def __init__(self, model, temperature):  # 构造函数
        super().__init__()
        self.__config__.extra = Extra.allow
        self.model = model
        self.temperature = temperature
        self.client = ZhipuAI()

    @property
    def _llm_type(self) -> str:
        return "zhipuai"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        print(f"\n----------------messages----------------\n")
        print(f"messages:{prompt}")
        print(f"\n----------------messages end----------------\n")
        response = self.client.chat.completions.create(
            model=self.model,  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": f"{prompt}"}
            ],
            temperature=self.temperature,
            stream=False
        )
        return response.choices[0].message.content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}