# flake8: noqa: E501
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str, float]


class Gemini(BaseAPIModel):
    """Model wrapper around Gemini models using the official google-genai SDK.

    Documentation: https://googleapis.github.io/python-genai/

    Args:
        path (str): The name of Gemini model.
            e.g. `gemini-2.5-flash`
        key (str): Authorization key. Set to 'ENV' to read from
            the GEMINI_API_KEY environment variable.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 2.
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retries if the API call fails. Defaults to 2.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'front', 'mid' and 'rear' represents the part
            of input to truncate. Defaults to 'none'.
        temperature (float, optional): Sampling temperature. If None,
            uses the API default.
        top_p (float, optional): Nucleus sampling parameter. If None,
            uses the API default.
        top_k (int, optional): Top-k sampling parameter. If None,
            uses the API default.
        api_base (str, optional): Custom API base URL for third-party
            proxies. If None, uses Google's official endpoint.
        token_count_key (str, optional): Authorization key used only for
            token counting. If set to 'ENV', reads
            GEMINI_TOKEN_COUNT_API_KEY first, then GEMINI_API_KEY.
            When unset and `api_base` is None, the generation key is reused.
        token_count_api_base (str, optional): Custom API base URL for token
            counting. If None, uses Google's official endpoint.
        token_count_path (str, optional): Model name used for token counting.
            If None, reuse `path`.
        thinking_config (dict, optional): Thinking configuration. Supports:
            - For Gemini 3.x: {'thinking_level': 'high'|'medium'|'low'|'minimal'}
            - For Gemini 2.5: {'thinking_budget': 1024}
            - To get thought summaries: {'include_thoughts': True}
            These can be combined, e.g.
            {'thinking_level': 'high', 'include_thoughts': True}
        token_safe_margin (int): Reserved token budget to absorb token counting
            differences and request overhead. Defaults to 256.
        request_timeout (int): Timeout for each HTTP request in seconds.
            Prevents the process from hanging indefinitely when the API
            endpoint is unresponsive. Defaults to 300 (5 minutes).
    """

    def __init__(
        self,
        key: str,
        path: str,
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        mode: str = 'none',
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        api_base: Optional[str] = None,
        token_count_key: Optional[str] = None,
        token_count_api_base: Optional[str] = None,
        token_count_path: Optional[str] = None,
        thinking_config: Optional[Dict] = None,
        token_safe_margin: int = 256,
        request_timeout: int = 300,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            meta_template=meta_template,
            retry=retry,
        )
        assert isinstance(key, str)
        if key == 'ENV':
            if 'GEMINI_API_KEY' not in os.environ:
                raise ValueError('GEMINI API key is not set.')
            key = os.getenv('GEMINI_API_KEY')
        if token_count_key == 'ENV':
            token_count_key = os.getenv('GEMINI_TOKEN_COUNT_API_KEY',
                                        os.getenv('GEMINI_API_KEY'))

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                'Please install the google-genai SDK: '
                'pip install google-genai')

        timeout_ms = request_timeout * 1000
        client_kwargs = {'api_key': key}
        if api_base:
            client_kwargs['http_options'] = types.HttpOptions(
                base_url=api_base, timeout=timeout_ms)
        else:
            client_kwargs['http_options'] = types.HttpOptions(
                timeout=timeout_ms)
        self.client = genai.Client(**client_kwargs)
        count_client_kwargs = None
        if token_count_key:
            count_client_kwargs = {'api_key': token_count_key}
        elif api_base is None:
            count_client_kwargs = {'api_key': key}

        self.count_client = None
        if count_client_kwargs is not None:
            count_http_opts = {'timeout': timeout_ms}
            if token_count_api_base:
                count_http_opts['base_url'] = token_count_api_base
            count_client_kwargs['http_options'] = types.HttpOptions(
                **count_http_opts)
            self.count_client = genai.Client(**count_client_kwargs)

        self.types = types
        assert mode in ['none', 'front', 'mid', 'rear']
        self.mode = mode
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.token_count_path = token_count_path or path
        self.thinking_config = thinking_config
        self.token_safe_margin = token_safe_margin
        self._count_tokens_disabled = self.count_client is None
        self._fallback_token_safe_margin = max(token_safe_margin * 4, 8192)

    def _build_thinking_config(self):
        """Build ThinkingConfig from the thinking_config dict."""
        if not self.thinking_config:
            return None
        types = self.types
        kwargs = {}
        if 'include_thoughts' in self.thinking_config:
            kwargs['include_thoughts'] = self.thinking_config['include_thoughts']
        if 'thinking_level' in self.thinking_config:
            level_map = {
                'minimal': types.ThinkingLevel.MINIMAL,
                'low': types.ThinkingLevel.LOW,
                'medium': types.ThinkingLevel.MEDIUM,
                'high': types.ThinkingLevel.HIGH,
            }
            kwargs['thinking_level'] = level_map[
                self.thinking_config['thinking_level'].lower()]
        if 'thinking_budget' in self.thinking_config:
            kwargs['thinking_budget'] = self.thinking_config['thinking_budget']
        return types.ThinkingConfig(**kwargs)

    def _build_messages(self, input: PromptType) -> tuple[Optional[str], List[Dict]]:
        """Convert OpenCompass prompt format into Gemini messages."""
        system_prompt = None

        if isinstance(input, str):
            return system_prompt, [{'role': 'user', 'text': input}]

        messages = []
        for item in input:
            if item['role'] == 'SYSTEM':
                system_prompt = item['prompt']
        for item in input:
            if item['role'] == 'HUMAN':
                messages.append({'role': 'user', 'text': item['prompt']})
            elif item['role'] == 'BOT':
                messages.append({'role': 'model', 'text': item['prompt']})
        return system_prompt, messages

    def _to_sdk_messages(self, messages: List[Dict]):
        """Convert internal message dicts to google-genai Content objects."""
        types = self.types
        return [
            types.Content(
                role=message['role'],
                parts=[types.Part.from_text(text=message['text'])],
            ) for message in messages
        ]

    def _fallback_token_count(self, messages: List[Dict],
                              system_prompt: Optional[str]) -> int:
        """Fallback rough token count when official count_tokens is unavailable."""
        total_tokens = sum(self.get_token_len(message['text'])
                           for message in messages)
        if system_prompt:
            total_tokens += self.get_token_len(system_prompt)
        return total_tokens

    def _is_permanent_count_error(self, error: Exception) -> bool:
        """Return whether a count_tokens failure is deterministic."""
        error_msg = str(error).lower()
        transient_markers = [
            'connection reset',
            'connection aborted',
            'connection refused',
            'timed out',
            'timeout',
            'temporarily unavailable',
            'service unavailable',
            'gateway time-out',
            'gateway timeout',
            'resource_exhausted',
            'rate limit',
            'too many requests',
            '429 ',
            ' 500',
            ' 502',
            ' 503',
            ' 504',
            '[errno 104]',
        ]
        permanent_markers = [
            'not supported for counttokens',
            'unsupported for counttokens',
            'counttokens. call listmodels',
            'api key not valid',
            'invalid api key',
            'permission_denied',
            'unauthorized',
            'authentication',
            '403 ',
            '404 ',
            'not_found',
            'not found',
        ]
        if any(marker in error_msg for marker in transient_markers):
            return False
        return any(marker in error_msg for marker in permanent_markers)

    def _get_effective_safe_margin(self, used_official_count: bool) -> int:
        """Use a larger margin when token count falls back to rough estimates."""
        if used_official_count:
            return self.token_safe_margin
        return self._fallback_token_safe_margin

    def _get_effective_input_budget(self, max_input_tokens: int,
                                    used_official_count: bool) -> int:
        """Shrink the usable budget when count_tokens falls back."""
        extra_margin = self._get_effective_safe_margin(
            used_official_count) - self.token_safe_margin
        return max(0, max_input_tokens - extra_margin)

    def _count_request_tokens(self, messages: List[Dict],
                              system_prompt: Optional[str]) -> tuple[int, bool]:
        """Count request tokens with the official Gemini SDK.

        Gemini Developer API currently rejects system_instruction in
        CountTokensConfig, so system prompts are counted separately as plain
        text instead of being passed through config.
        """
        if self._count_tokens_disabled:
            return self._fallback_token_count(messages, system_prompt), False

        last_error = None
        for count_retry_idx in range(self.retry):
            self.wait()
            try:
                total_tokens = 0
                if messages:
                    response = self.count_client.models.count_tokens(
                        model=self.token_count_path,
                        contents=self._to_sdk_messages(messages),
                    )
                    if response.total_tokens is None:
                        raise ValueError(
                            'count_tokens returned no total_tokens.')
                    total_tokens += response.total_tokens

                if system_prompt:
                    if messages:
                        self.wait()
                    system_response = self.count_client.models.count_tokens(
                        model=self.token_count_path,
                        contents=system_prompt,
                    )
                    if system_response.total_tokens is None:
                        raise ValueError(
                            'count_tokens returned no total_tokens for '
                            'system prompt.')
                    total_tokens += system_response.total_tokens

                return total_tokens, True
            except Exception as e:
                last_error = e
                if self._is_permanent_count_error(e):
                    self._count_tokens_disabled = True
                    self.logger.warning(
                        'Gemini count_tokens is unavailable for this model or '
                        'credential (%s). Permanently falling back to the '
                        'rough BaseAPIModel token estimate.', e)
                    break
                if count_retry_idx < self.retry - 1:
                    backoff = count_retry_idx + 1
                    self.logger.warning(
                        'Gemini count_tokens transient failure (%s). '
                        'Retrying in %ss (%s/%s).', e, backoff,
                        count_retry_idx + 1, self.retry)
                    time.sleep(backoff)
                    continue
                self.logger.warning(
                    'Gemini count_tokens transient failure (%s). Using the '
                    'rough BaseAPIModel token estimate for this request.',
                    e)
        return self._fallback_token_count(messages, system_prompt), False

    def _slice_text(self, text: str, keep_chars: int, mode: str) -> str:
        """Slice text by character budget using the configured truncation mode."""
        if keep_chars <= 0:
            return ''
        if keep_chars >= len(text):
            return text
        if mode == 'front':
            return text[-keep_chars:]
        if mode == 'rear':
            return text[:keep_chars]

        head_chars = keep_chars // 2
        tail_chars = keep_chars - head_chars
        return text[:head_chars] + text[-tail_chars:]

    def _trim_text_to_fit(self, text: str, messages: List[Dict], index: int,
                          system_prompt: Optional[str], max_input_tokens: int,
                          mode: str) -> str:
        """Trim a single message until the whole request fits the token budget."""
        if not text:
            return text

        left, right = 0, len(text)
        best = ''
        while left <= right:
            mid = (left + right) // 2
            candidate = self._slice_text(text, mid, mode)
            trial_messages = [message.copy() for message in messages]
            trial_messages[index]['text'] = candidate
            token_count, used_official_count = self._count_request_tokens(
                trial_messages, system_prompt)
            effective_budget = self._get_effective_input_budget(
                max_input_tokens, used_official_count)
            if token_count <= effective_budget:
                best = candidate
                left = mid + 1
            else:
                right = mid - 1
        return best

    def _trim_system_prompt_to_fit(self, system_prompt: str, messages: List[Dict],
                                   max_input_tokens: int, mode: str) -> str:
        """Trim the system prompt as a last resort."""
        if not system_prompt:
            return system_prompt

        left, right = 0, len(system_prompt)
        best = ''
        while left <= right:
            mid = (left + right) // 2
            candidate = self._slice_text(system_prompt, mid, mode)
            token_count, used_official_count = self._count_request_tokens(
                messages, candidate)
            effective_budget = self._get_effective_input_budget(
                max_input_tokens, used_official_count)
            if token_count <= effective_budget:
                best = candidate
                left = mid + 1
            else:
                right = mid - 1
        return best

    def _truncate_messages_to_budget(
        self,
        messages: List[Dict],
        system_prompt: Optional[str],
        max_input_tokens: int,
        mode: str,
    ) -> tuple[List[Dict], Optional[str]]:
        """Trim history first and keep the latest user turn as much as possible."""
        processed_messages = [message.copy() for message in messages]
        trim_order = list(range(max(len(processed_messages) - 1, 0)))
        if processed_messages:
            trim_order.append(len(processed_messages) - 1)

        for index in trim_order:
            current_tokens, used_official_count = self._count_request_tokens(
                processed_messages, system_prompt)
            effective_budget = self._get_effective_input_budget(
                max_input_tokens, used_official_count)
            if current_tokens <= effective_budget:
                return processed_messages, system_prompt
            processed_messages[index]['text'] = self._trim_text_to_fit(
                processed_messages[index]['text'],
                processed_messages,
                index,
                system_prompt,
                max_input_tokens,
                mode,
            )

        current_tokens, used_official_count = self._count_request_tokens(
            processed_messages, system_prompt)
        effective_budget = self._get_effective_input_budget(
            max_input_tokens, used_official_count)
        if current_tokens > effective_budget and system_prompt:
            system_prompt = self._trim_system_prompt_to_fit(
                system_prompt, processed_messages, max_input_tokens, mode)

        return processed_messages, system_prompt

    def _preprocess_messages(
        self,
        input: PromptType,
        max_out_len: int,
    ) -> tuple[Optional[str], List, int]:
        """Count input tokens, truncate if needed, and adjust output budget."""
        system_prompt, messages = self._build_messages(input)
        max_input_tokens = self.max_seq_len - max_out_len - self.token_safe_margin
        if max_input_tokens <= 0:
            raise ValueError(
                f'max_out_len ({max_out_len}) leaves no room for input tokens '
                f'under max_seq_len ({self.max_seq_len}).')

        input_tokens, used_official_count = self._count_request_tokens(
            messages, system_prompt)
        effective_budget = self._get_effective_input_budget(
            max_input_tokens, used_official_count)
        truncated = False
        if input_tokens > effective_budget:
            if self.mode == 'none':
                raise ValueError(
                    f'Input length ({input_tokens}) exceeds max input budget '
                    f'({effective_budget}). Please either change the mode or '
                    f'increase max_seq_len.')
            messages, system_prompt = self._truncate_messages_to_budget(
                messages, system_prompt, max_input_tokens, self.mode)
            input_tokens, used_official_count = self._count_request_tokens(
                messages, system_prompt)
            effective_budget = self._get_effective_input_budget(
                max_input_tokens, used_official_count)
            truncated = True
            if input_tokens > effective_budget:
                raise ValueError(
                    f'Input length ({input_tokens}) still exceeds max input '
                    f'budget ({effective_budget}) after truncation.')
        if truncated or not used_official_count:
            count_source = 'official count_tokens' if used_official_count \
                else 'rough fallback estimate'
            self.logger.info(
                'Gemini request budgeting used %s: input_tokens=%s, '
                'effective_budget=%s, max_seq_len=%s, max_out_len=%s, '
                'truncated=%s.', count_source, input_tokens,
                effective_budget, self.max_seq_len, max_out_len, truncated)

        original_max_out_len = max_out_len
        effective_safe_margin = self._get_effective_safe_margin(
            used_official_count)
        max_out_len = min(max_out_len,
                          self.max_seq_len - input_tokens -
                          effective_safe_margin)
        if max_out_len <= 0:
            raise ValueError(
                f'max_out_len ({max_out_len}) is less than or equal to 0. '
                f'This may be due to input length ({input_tokens}) being too '
                f'close to max_seq_len ({self.max_seq_len}).')
        if max_out_len < original_max_out_len:
            self.logger.warning(
                'max_out_len was truncated from %s to %s due to input length '
                '(count_source=%s, safe_margin=%s).',
                original_max_out_len, max_out_len,
                'official' if used_official_count else 'fallback',
                effective_safe_margin)

        return system_prompt, self._to_sdk_messages(messages), max_out_len

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs)))
        self.flush()
        return results

    def _generate(
        self,
        input: PromptType,
        max_out_len: int = 512,
    ) -> str:
        """Generate results given an input.

        Args:
            input (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        types = self.types
        system_prompt, messages, max_out_len = self._preprocess_messages(
            input, max_out_len)

        # Build generation config - only set parameters that are explicitly
        # configured, leaving others as None so the API uses its defaults.
        config_kwargs = {}
        if system_prompt is not None:
            config_kwargs['system_instruction'] = system_prompt
        if self.temperature is not None:
            config_kwargs['temperature'] = self.temperature
        if self.top_p is not None:
            config_kwargs['top_p'] = self.top_p
        if self.top_k is not None:
            config_kwargs['top_k'] = self.top_k

        config_kwargs['max_output_tokens'] = max_out_len

        thinking_cfg = self._build_thinking_config()
        if thinking_cfg is not None:
            config_kwargs['thinking_config'] = thinking_cfg

        config = types.GenerateContentConfig(**config_kwargs)

        for _ in range(self.retry):
            self.wait()
            try:
                response = self.client.models.generate_content(
                    model=self.path,
                    contents=messages,
                    config=config,
                )
            except Exception as e:
                self.logger.error(f'API error: {e}')
                time.sleep(1)
                continue

            # Extract answer text, skipping thinking parts
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.finish_reason and 'SAFETY' in str(
                        candidate.finish_reason):
                    return ("Due to Google's restrictive policies, "
                            "I am unable to respond to this question.")
                if candidate.content and candidate.content.parts:
                    thought_parts = []
                    answer_parts = []
                    for part in candidate.content.parts:
                        if not part.text:
                            continue
                        if getattr(part, 'thought', False):
                            thought_parts.append(part.text)
                        else:
                            answer_parts.append(part.text)
                    result = ''
                    if thought_parts:
                        result += '<think>\n' + '\n'.join(thought_parts) + '\n</think>\n'
                    if answer_parts:
                        result += '\n'.join(answer_parts)
                    if result:
                        return result.strip()

            self.logger.error(f'Empty response: {response}')
            return ''

        raise RuntimeError('API call failed.')
