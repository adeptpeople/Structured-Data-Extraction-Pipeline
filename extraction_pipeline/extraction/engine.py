from __future__ import annotations

import json
from typing import Optional

import openai

from extraction_pipeline import config
from extraction_pipeline.extraction.prompts import SYSTEM_PROMPT, RETRY_PROMPT_TEMPLATE
from extraction_pipeline.extraction.tool_schema import EXTRACTION_TOOL_DEFINITION


class ExtractionEngine:
    """Wraps a single OpenAI Responses API tool-use call."""

    def __init__(self, client: Optional[openai.AsyncOpenAI] = None) -> None:
        self._client = client or openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    async def extract(
        self,
        document_id: str,
        document_text: str,
        few_shot_messages: list[dict],
        error_feedback: Optional[str] = None,
        attempt_number: int = 1,
    ) -> dict:
        """
        Call the OpenAI Responses API with strict tool-use.
        Returns the raw parsed dict from the tool call arguments.
        """
        if error_feedback and attempt_number > 1:
            user_content = RETRY_PROMPT_TEMPLATE.format(
                original_document_content=f"DOCUMENT ID: {document_id}\n\n{document_text}",
                attempt_number=attempt_number,
                max_retries=config.MAX_RETRIES,
                error_list=error_feedback,
            )
        else:
            user_content = f"DOCUMENT ID: {document_id}\n\n{document_text}"

        messages = few_shot_messages + [{"role": "user", "content": user_content}]

        response = await self._client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=[EXTRACTION_TOOL_DEFINITION],
            tool_choice={"type": "function", "function": {"name": "extract_document"}},
            temperature=0.0,
        )

        tool_call = response.choices[0].message.tool_calls[0]
        raw_dict = json.loads(tool_call.function.arguments)
        return raw_dict

    def build_batch_request_body(
        self,
        document_id: str,
        document_text: str,
        few_shot_messages: list[dict],
    ) -> dict:
        """Build the request body for use in a Message Batches API request."""
        user_content = f"DOCUMENT ID: {document_id}\n\n{document_text}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += few_shot_messages
        messages += [{"role": "user", "content": user_content}]

        return {
            "model": config.OPENAI_BATCH_MODEL,
            "messages": messages,
            "tools": [EXTRACTION_TOOL_DEFINITION],
            "tool_choice": {"type": "function", "function": {"name": "extract_document"}},
            "temperature": 0.0,
        }
