import pytest

from rodiumai.resources.chat import Chat, ChatCompletion, ChatCompletionChunk, Choice, Message, CompletionUsage


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_standard_response_parsed_correctly(self):
        pass

    def test_empty_message_list_raises_value_error(self):
        import pytest
        chat = Chat(None)
        with pytest.raises(ValueError, match="messages must not be empty"):
            chat.completions.create(messages=[])

    def test_temperature_out_of_range_raises_value_error(self):
        chat = Chat(None)
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            chat.completions.create(messages=[{"role": "user", "content": "hi"}], temperature=3.0)

    def test_temperature_below_zero_raises_value_error(self):
        chat = Chat(None)
        with pytest.raises(ValueError):
            chat.completions.create(messages=[{"role": "user", "content": "hi"}], temperature=-1.0)
