from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Callback for when the LLM starts running"""
        print(f"***Prompt to LLM was sent: {prompts[0]}***")
        print("*********************************************")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Callback for when the LLM finishes running"""
        print(f"***LLM response was received: {response}***")
        print("*********************************************")
