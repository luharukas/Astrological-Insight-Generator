"""
Wrapper classes for ChatGPT and Gemini LLMs using direct OpenAI and Google GenAI libraries.
"""

import base64
import json
import os
from typing import Any, List, Literal, Optional, Type, TypeVar, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types
from loguru import logger
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from .llm_utils import image_path_to_b64_string

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# Generic type for pydantic output
T = TypeVar("T", bound=BaseModel)


# Main OpenAI client wrapper
class _OpenAIClientWrapper:
    def __init__(self) -> None:
        self.available_models: List[str] = ["gpt-5-mini", "gpt-4.1-mini"]
        self.client = OpenAI(api_key=OPENAI_API_KEY, max_retries=1)
        self.model_name: str = self.available_models[0]
        self.temperature: float = 0.0
        self.top_p: float = 0.1
        logger.debug(f" Initialized OpenAIClientWrapper with default model '{self.model_name}'")

    def invoke(
        self,
        prompt: str,
        output_pydantic: Type[T],
        system_prompt: str = "You are a helpful assistant.",
        image_paths: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> T:
        """
        Call the OpenAI API and parse the response into a Pydantic model.

        Steps:
          1. Construct chat messages with system and user prompts.
          2. Attach image data URLs if provided.
          3. Add Pydantic JSON schema instruction to system prompt.
          4. Send request via OpenAI client and retrieve raw response.
          5. Extract JSON payload from response and validate with Pydantic.

        Args:
            prompt (str): The main user prompt text.
            output_pydantic (Type[T]): Pydantic model class for parsing the output.
            system_prompt (str): The system-level instructions.
            image_paths (Optional[List[str]]): Paths to images to include.
            model_name (Optional[str]): Override default model name.
            temperature (Optional[float]): Sampling temperature.
            top_p (Optional[float]): Sampling top-p probability.

        Returns:
            T: Parsed Pydantic model instance of the LLM response.
        """

        logger.debug(
            f" OpenAI invoke called: model_name={model_name or self.model_name}, prompt_length={len(prompt)}"
        )
        user_content: List[ChatCompletionContentPartParam] = [{"type": "text", "text": prompt}]

        if image_paths:
            for path in image_paths:
                b64_image = image_path_to_b64_string(path)
                user_content.append({"type": "image_url", "image_url": {"url": b64_image}})

        messages: List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = (
            self.client.chat.completions.parse(
                model=(
                    model_name
                    if (model_name is not None) and (model_name in self.available_models)
                    else self.model_name
                ),
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                top_p=top_p if top_p is not None else self.top_p,
                seed=42,
                response_format=output_pydantic,
            )
            .choices[0]
            .message.content
        )

        logger.debug(
            f" OpenAI raw response received (truncated to 200 chars): {response[:200] if response else None}"
        )
        if response is None:
            logger.error(" No content in the response.")
            raise ValueError("OpenAI response contained no message content.")
        else:
            try:
                parsed_response = output_pydantic(**json.loads(response))
                return parsed_response
            except Exception as e:
                logger.error(f" Failed to parse or validate response: {e}")
                raise

    def list_available_models(self) -> List[str]:
        """
        Retrieve the list of available OpenAI model names.

        Steps:
          1. Access the internal `available_models` list.
          2. Log and return the list.

        Returns:
            List[str]: Model identifiers available for use.
        """
        logger.debug(f" OpenAI available models: {self.available_models}")
        return self.available_models


# Main Gemini (Google) client wrapper
class _GeminiClientWrapper:
    def __init__(self) -> None:
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.available_models: List[str] = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
        self.model_name: str = self.available_models[0]
        self.temperature: float = 0.0
        self.top_p: float = 0.1
        logger.debug(f" Initialized GeminiClientWrapper with default model '{self.model_name}'")

    def invoke(
        self,
        prompt: str,
        output_pydantic: Type[T],
        system_prompt: str = "You are a helpful assistant.",
        image_paths: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> T:
        """
        Call the Google Gemini API and parse the response into a Pydantic model.

        Steps:
          1. Build message parts from prompt and images.
          2. Append Pydantic JSON schema instructions to system prompt.
          3. Configure generation settings with safety disabled.
          4. Invoke Gemini client and retrieve raw text.
          5. Extract JSON payload and validate with Pydantic.

        Returns:
            T: Parsed Pydantic model instance of the LLM response.
        """

        # Build message parts: prompt and optional images
        # Prepare message parts: primary prompt string and optional images

        logger.debug(
            f" Gemini invoke called: model_name={model_name or self.model_name}, prompt_length={len(prompt)}"
        )
        message_parts: List[Any] = [types.Part.from_text(text=prompt)]  # Primary text prompt

        if image_paths:
            for path in image_paths:
                b64_image = image_path_to_b64_string(path)
                message_parts.append(
                    types.Part.from_bytes(
                        mime_type=b64_image.split(";")[0].split(":")[1],  # Extract MIME type
                        data=base64.b64decode(b64_image.split(";base64,")[1]),
                    )
                )

        # Build generation config with safety disabled
        # Config dict for GenerateContentConfigOrDict, disabling safety filters
        # Build config dict for generation, disabling safety filters
        config: types.GenerateContentConfigDict = {
            "system_instruction": system_prompt,
            # The model is prompted to output JSON; parsing will be handled in wrapper
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p if top_p is not None else self.top_p,
            # Disable all safety filters by setting thresholds to OFF
            "safety_settings": [
                {
                    "category": types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": types.HarmBlockThreshold.OFF,
                },
                {
                    "category": types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": types.HarmBlockThreshold.OFF,
                },
                {
                    "category": types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": types.HarmBlockThreshold.OFF,
                },
                {
                    "category": types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": types.HarmBlockThreshold.OFF,
                },
                {
                    "category": types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                    "threshold": types.HarmBlockThreshold.OFF,
                },
            ],
            "seed": 42,
            "response_mime_type": "application/json",
            "response_schema": output_pydantic,
        }

        # Select model
        selected_model = model_name if model_name is not None else self.model_name

        # Create chat session and send message
        raw_response = self.client.models.generate_content(
            model=selected_model, contents=message_parts, config=config
        )

        logger.debug(
            f" Gemini raw response received (truncated to 200 chars): {raw_response.text[:200] if raw_response.text else None}"
        )

        if raw_response.parsed and isinstance(raw_response.parsed, output_pydantic):
            return raw_response.parsed
        else:
            logger.error(" Parsed data doesnot received.")
            raise
        # End of invoke method

    def list_available_models(self) -> List[str]:
        """
        Retrieve the list of available Gemini model names.

        Steps:
          1. Access the internal `available_models` list.
          2. Log and return the list.

        Returns:
            List[str]: Model identifiers available for use.
        """
        logger.debug(f" Gemini available models: {self.available_models}")
        return self.available_models


gemini = _GeminiClientWrapper()
openai = _OpenAIClientWrapper()


def invoke_llm_with_fallback(
    prompt: str,
    output_pydantic: Type[T],
    system_prompt: str = "You are a helpful assistant.",
    image_paths: Optional[List[str]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    provider: Literal["gemini", "openai", "all"] = "all",
) -> T:
    """
    Invoke LLM models with fallback strategy across Gemini and OpenAI.

    Steps:
      1. Attempt models from Gemini if requested.
      2. Attempt models from OpenAI if requested or after Gemini failures.
      3. Log each attempt and model outcome.
      4. Return the first successful parsed response or raise the last exception.

    Args:
        prompt (str): The user prompt to send.
        output_pydantic (Type[T]): Pydantic model class for parsing.
        system_prompt (str): System-level instructions.
        image_paths (Optional[List[str]]): Paths to images to include.
        temperature (Optional[float]): Sampling temperature.
        top_p (Optional[float]): Sampling top-p probability.
        provider (str): Provider selection: 'gemini', 'openai', or 'all'.

    Returns:
        T: Parsed Pydantic model instance.

    Raises:
        Exception: Last exception encountered if all models fail.
    """
    last_exception: Optional[Exception] = None

    logger.info(f" invoke_llm_with_fallback called with provider='{provider}'")
    # Try Gemini models if requested
    if provider in ("gemini", "all"):
        for model in gemini.available_models:
            try:
                logger.debug(f" Trying Gemini model {model}")
                return gemini.invoke(
                    prompt=prompt,
                    output_pydantic=output_pydantic,
                    system_prompt=system_prompt,
                    image_paths=image_paths,
                    model_name=model,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as e:
                logger.warning(f" Gemini model {model} failed: {e}")
                last_exception = e

    # Try OpenAI models if requested
    if provider in ("openai", "all"):
        for model in openai.available_models:
            try:
                logger.debug(f" Trying OpenAI model {model}")
                return openai.invoke(
                    prompt=prompt,
                    output_pydantic=output_pydantic,
                    system_prompt=system_prompt,
                    image_paths=image_paths,
                    model_name=model,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as e:
                logger.warning(f" OpenAI model {model} failed: {e}")
                last_exception = e

    # All attempts failed; raise the last exception
    if last_exception:
        raise last_exception
    raise RuntimeError(f"No LLM models were available for invocation with provider '{provider}'.")
