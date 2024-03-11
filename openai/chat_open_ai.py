from typing import (
    Any,
    Iterator,
    List,
    Optional,
    cast,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_openai import ChatOpenAI


class MyChatOpenAI(ChatOpenAI):
    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessageChunk]:
        yield cast(
            BaseMessageChunk, self.invoke(input, config=config, stop=stop, **kwargs)
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **({"stream": stream} if stream is not None else {}),
            **kwargs,
        }
        print(f"\n----------------messages----------------\n")
        print(f"\nmessages:{message_dicts}\n")
        print(f"\n----------------messages end----------------\n")
        response = self.client.create(messages=message_dicts, **params)
        print(f"\n----------------response----------------\n")
        print(f"\nresponse:{response}\n")
        print(f"\n----------------response end----------------\n")
        return self._create_chat_result(response)