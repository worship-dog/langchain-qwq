"""Test ChatQwQ chat model."""

from typing import Type

from dotenv import load_dotenv
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_qwq_modification.chat_models import ChatQwQ

load_dotenv()


class TestChatQwQIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatQwQ]:
        return ChatQwQ

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "qwq-plus",
        }

    @property
    def has_tool_choice(self) -> bool:
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        return False
