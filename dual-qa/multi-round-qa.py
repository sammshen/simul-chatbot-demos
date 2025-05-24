import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Dict
import random
import os
import glob

import openai
import pandas as pd
import numpy as np

from utils import AsyncLoopWrapper, init_logger

logger = init_logger(__name__, logging.INFO)

# Path to context database directory
CONTEXT_DB_PATH = "../contexts"

# Automatically scan for context files
def get_context_files():
    # Get all .txt, .pdf, and .html files in the contexts directory
    txt_files = glob.glob(os.path.join(CONTEXT_DB_PATH, "*.txt"))
    pdf_files = glob.glob(os.path.join(CONTEXT_DB_PATH, "*.pdf"))
    html_files = glob.glob(os.path.join(CONTEXT_DB_PATH, "*.html"))

    # Return just the filenames
    return [os.path.basename(f) for f in txt_files + pdf_files + html_files]

# Get available context files
CONTEXT_FILES = get_context_files()

# Log available files
logger.info(f"Found {len(CONTEXT_FILES)} files in contexts directory: {CONTEXT_FILES}")

# Cache for loaded file contents
CONTEXT_FILE_CONTENTS = {}

def load_context_file(filename):
    """Load a file from the contexts directory."""
    if filename in CONTEXT_FILE_CONTENTS:
        return CONTEXT_FILE_CONTENTS[filename]

    try:
        file_path = os.path.join(CONTEXT_DB_PATH, filename)
        if not os.path.exists(file_path):
            logger.error(f"Context file {filename} does not exist at {file_path}")
            return f"Error: {filename} documentation not found."

        # Handle based on file extension
        if filename.endswith('.txt'):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        elif filename.endswith('.pdf'):
            try:
                import PyPDF2
                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = ""
                    for page_num in range(len(pdf_reader.pages)):
                        content += pdf_reader.pages[page_num].extract_text() + "\n\n"
            except ImportError:
                logger.error("PyPDF2 not installed. Cannot read PDF files.")
                return f"Error: PyPDF2 not installed. Cannot read {filename}."
        elif filename.endswith('.html'):
            try:
                from bs4 import BeautifulSoup
                with open(file_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    # Extract text from HTML, removing script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()
                    content = soup.get_text(separator='\n')
                    # Remove multiple newlines
                    content = '\n'.join([line.strip() for line in content.split('\n') if line.strip()])
            except ImportError:
                logger.error("BeautifulSoup not installed. Cannot read HTML files.")
                return f"Error: BeautifulSoup not installed. Cannot read {filename}."
        else:
            logger.error(f"Unsupported file format for {filename}")
            return f"Error: {filename} has unsupported format."

        # Cache the content
        CONTEXT_FILE_CONTENTS[filename] = content
        logger.info(f"Loaded context file {filename} ({len(content)} bytes)")
        return content
    except Exception as e:
        logger.error(f"Failed to load context file {filename}: {e}")
        return f"Error loading documentation for {filename}: {str(e)}"

@dataclass
class WorkloadConfig:
    # Max number of users in the system concurrently
    num_users: int

    # Length of shared system prompt
    system_prompt_len: int

    # Length of the user-specific data
    user_info_len: int

    # Length of the answer in one round
    answer_len: int

    # Number of rounds in the conversation
    num_rounds: int

    # Overall QPS
    qps: int

    # Model name
    model: str

    # Whether to include user id in request header
    enable_user_id: bool


@dataclass
class UserConfig:
    # User id
    user_id: int

    # System prompt length
    system_prompt_len: int

    # Length of the user-specific data
    user_info_len: int

    # Answer length
    answer_len: int

    # Gap between two requests
    gap_between_requests: int

    # Num rounds
    num_rounds: int

    # Whether to include user id in request header
    enable_user_id: bool

    @staticmethod
    def new_user_config(user_id: int, workload_config: WorkloadConfig) -> "UserConfig":
        return UserConfig(
            user_id=user_id,
            system_prompt_len=workload_config.system_prompt_len,
            user_info_len=workload_config.user_info_len,
            answer_len=workload_config.answer_len,
            gap_between_requests=workload_config.num_users / workload_config.qps,
            num_rounds=workload_config.num_rounds,
            enable_user_id=workload_config.enable_user_id,
        )


class ChatHistory:

    def __init__(
        self,
    ):
        self.history = []

    def on_user_query(self, query: str):
        if len(self.history) == 0:
            self.history.append({"role": "user", "content": query})
        else:
            assert self.history[-1]["role"] == "assistant", "Expect system response"
            self.history.append({"role": "user", "content": query})

    def on_system_response(self, response: str):
        assert len(self.history) > 0, "Expect user query"
        assert self.history[-1]["role"] == "user", "Expect user query"
        self.history.append({"role": "assistant", "content": response})

    def get_messages_for_openai(self):
        return self.history

    def __len__(self):
        return len(self.history)


@dataclass
class Response:
    body: str
    ttft: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int
    launch_time: float
    finish_time: float
    # Add endpoint identification
    endpoint_name: str = "default"


class RequestExecutor:

    def __init__(self, base_url: str, model: str, endpoint_name: str = "default"):
        # Ensure base_url ends with /v1
        if not base_url.endswith('/v1'):
            base_url = base_url.rstrip('/') + '/v1'

        # For vLLM server, we don't need an API key, but the client requires one
        self.client = openai.AsyncOpenAI(
            api_key="EMPTY",  # Dummy API key for vLLM server
            base_url=base_url
        )
        self.model = model
        self.endpoint_name = endpoint_name
        logging.info(f"Initialized OpenAI client {endpoint_name} with base_url={base_url} and model={model}")
        self.loop = AsyncLoopWrapper.GetOrStartLoop()
        self.request_history = []

    async def _async_launch_request(self, messages: List[Dict[str, str]],  max_tokens: int,
                                    extra_headers: Optional[Dict[str, str]] = None):
        try:
            logging.info(f"Sending request to endpoint {self.endpoint_name}, model {self.model} with messages: {messages}")

            # Initialize response tracking variables
            words = ""
            tokens_out = 0
            tokens_prefill = 0
            start_time = time.time()
            first_token_time = None

            # Use chat.completions API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=max_tokens,
                temperature=0.0,
                stream_options={"include_usage": True},
                extra_headers=extra_headers,
                seed=42,
            )

            # Process the streaming response
            async for chunk in response:
                if not chunk.choices:
                    continue

                # Handle content
                if chunk.choices[0].delta.content is not None:
                    if first_token_time is None and chunk.choices[0].delta.content != "":
                        first_token_time = time.time()
                    words += chunk.choices[0].delta.content

            # Handle token counts if available
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                tokens_out = chunk.usage.completion_tokens
                tokens_prefill = chunk.usage.prompt_tokens

            # If we didn't get token counts from streaming, try to get them from the final response
            if tokens_out == 0 or tokens_prefill == 0:
                print("No token counts from streaming, getting final response")
                print(f"{tokens_out}, {tokens_prefill}")
                try:
                    if use_chat_completions:
                        final_response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            stream=False,
                            temperature=0.0,
                            seed=42,
                        )
                    else:
                        prompt = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in messages])
                        final_response = await self.client.completions.create(
                            model=self.model,
                            prompt=prompt,
                            stream=False,
                            temperature=0.0,
                            seed=42,
                        )

                    if hasattr(final_response, 'usage') and final_response.usage is not None:
                        tokens_out = final_response.usage.completion_tokens
                        tokens_prefill = final_response.usage.prompt_tokens
                except Exception as e:
                    logging.warning(f"Failed to get token counts from final response: {e}")

            # # Calculate timing metrics
            ttft = first_token_time - start_time if first_token_time else 0
            generation_time = time.time() - first_token_time if first_token_time else 0

            return Response(
                body=words,
                ttft=ttft,
                generation_time=generation_time,
                prompt_tokens=tokens_prefill,
                generation_tokens=tokens_out,
                launch_time=start_time,
                finish_time=time.time(),
                endpoint_name=self.endpoint_name
            )

        except Exception as e:
            logging.error(f"Error in _async_launch_request: {str(e)}")
            logging.error(f"Request details - endpoint: {self.endpoint_name}, model: {self.model}, messages: {messages}")
            raise

    def launch_request(
        self,
        chat_history: ChatHistory,
        max_tokens: int,
        finish_callback,
        extra_headers=None,
    ):
        """
        finish_callback: Callable[[Response], None]
        """
        messages = chat_history.get_messages_for_openai()
        real_callback = lambda x: finish_callback(x.result())
        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(messages, max_tokens, extra_headers), self.loop
        )
        future.add_done_callback(real_callback)


class UserSession:

    def __init__(self, user_config: UserConfig, use_sharegpt=False, sharegpt_data=None, real_time_stats_file=None):
        self.user_config = user_config
        self.last_request_time = None
        self.chat_history = ChatHistory()
        self.question_id = 0
        self.use_sharegpt = use_sharegpt
        if self.use_sharegpt:
            self.sharegpt_data = sharegpt_data
            if self.sharegpt_data["num_round"] % 2 == 0:
                self.start_with_gpt = False
            else:
                self.start_with_gpt = True

        # Initialize with our known endpoints
        self.has_unfinished_request = {
            "ProductionStack": False,
            "Other": False
        }
        self.last_unfinished_log = 0

        # Track metrics per endpoint
        self.endpoints = {}

        # Track current documentation files
        self.current_docs = {
            "system": None,
            "user": None
        }

        self.finished = False

        self.real_time_stats_file = real_time_stats_file

    def _update_result(self, response: Response):
        endpoint = response.endpoint_name
        if endpoint not in self.endpoints:
            self.endpoints[endpoint] = {
                "prompt_lengths": [],
                "generation_lengths": [],
                "ttfts": [],
                "generation_times": [],
                "launch_times": [],
                "finish_times": []
            }

        self.endpoints[endpoint]["prompt_lengths"].append(response.prompt_tokens)
        self.endpoints[endpoint]["generation_lengths"].append(response.generation_tokens)
        self.endpoints[endpoint]["ttfts"].append(response.ttft)
        self.endpoints[endpoint]["generation_times"].append(response.generation_time)
        self.endpoints[endpoint]["launch_times"].append(response.launch_time)
        self.endpoints[endpoint]["finish_times"].append(response.finish_time)

    def _build_system_prompt(self):
        """Build a system prompt using content from context files."""
        # For deterministic behavior based on user id
        seed = self.user_config.user_id
        rand = random.Random(seed)

        # Check if we have any files
        if not CONTEXT_FILES:
            logger.error("No context files found. Please add files to the contexts directory.")
            return "You are a helpful AI assistant. No context documents were found to assist with queries."

        # Select a file for the system prompt
        system_file = rand.choice(CONTEXT_FILES)
        system_content = load_context_file(system_file)

        # Calculate approximately how many tokens we need
        # (rough estimate: 4 chars = 1 token)
        target_chars = self.user_config.system_prompt_len * 4

        # Make sure we have enough content to work with
        if len(system_content) > target_chars:
            # Extract a coherent chunk by finding paragraph boundaries
            paragraphs = system_content.split("\n\n")
            chunk = ""
            selected_paragraphs = []

            # Start from a random paragraph
            start_idx = rand.randint(0, max(0, len(paragraphs) - 5))

            # Add paragraphs until we reach our target size
            for i in range(start_idx, len(paragraphs)):
                if len(chunk) + len(paragraphs[i]) + 2 <= target_chars:
                    selected_paragraphs.append(paragraphs[i])
                    chunk += paragraphs[i] + "\n\n"
                else:
                    # If we can't fit a whole paragraph, add as much as we can
                    if not selected_paragraphs:
                        chunk = system_content[:target_chars]
                    break

            system_content = chunk.strip()

        # Select a different file for user-specific content
        user_files = [f for f in CONTEXT_FILES if f != system_file]
        user_file = rand.choice(user_files) if user_files else system_file
        user_content = load_context_file(user_file)

        # Calculate approximately how many tokens we need for user content
        target_chars_user = self.user_config.user_info_len * 4

        # Make sure we have enough content to work with
        if len(user_content) > target_chars_user:
            # Extract a coherent chunk by finding paragraph boundaries
            paragraphs = user_content.split("\n\n")
            chunk = ""
            selected_paragraphs = []

            # Start from a random paragraph
            start_idx = rand.randint(0, max(0, len(paragraphs) - 5))

            # Add paragraphs until we reach our target size
            for i in range(start_idx, len(paragraphs)):
                if len(chunk) + len(paragraphs[i]) + 2 <= target_chars_user:
                    selected_paragraphs.append(paragraphs[i])
                    chunk += paragraphs[i] + "\n\n"
                else:
                    # If we can't fit a whole paragraph, add as much as we can
                    if not selected_paragraphs:
                        chunk = user_content[:target_chars_user]
                    break

            user_content = chunk.strip()

        # Build the system prompt
        system_prompt = (
            f"You are a helpful AI assistant. "
            f"Here's information from the document '{system_file}' that you should use as context:\n\n"
            f"{system_content}\n\n"
            f"For user {self.user_config.user_id}, "
            f"here is additional information from the document '{user_file}' that they might ask about:\n\n"
            f"{user_content}"
        )

        # Store the documentation files used
        self.current_docs = {
            "system": system_file,
            "user": user_file
        }

        # Log which documents we're using
        logger.debug(f"User {self.user_config.user_id} using documents: {system_file} and {user_file}")

        return system_prompt

    def _build_new_question(self):
        """Build a more realistic question based on the available documents."""
        self.question_id += 1

        # For deterministic behavior based on user id and question id
        seed = self.user_config.user_id * 100 + self.question_id
        rand = random.Random(seed)

        # Create different question templates for variety
        question_templates = [
            "Question #{}: Can you explain the key concepts in the '{}'? I'd also like to know how it relates to the information in '{}'.",

            "Question #{}: I'm studying the document '{}' and I need a detailed explanation of its main points. Could you also compare it with the concepts in '{}'?",

            "Question #{}: What are the most important points in '{}'? And how might they connect with the information from '{}'?",

            "Question #{}: Can you analyze the document '{}' and provide a detailed summary? I'm also interested in how it might relate to '{}'.",

            "Question #{}: I need to understand '{}' in depth. Could you explain its key elements and how they might connect with concepts from '{}'?",

            "Question #{}: Could you provide an in-depth analysis of '{}' and explain how it might complement or contrast with the information in '{}'?",

            "Question #{}: What are the core principles described in '{}'? Also, how do these principles compare to those in '{}'?"
        ]

        # Select a template and format it
        template = rand.choice(question_templates)

        # Randomly decide which document to focus on first
        if rand.random() < 0.5:
            return template.format(
                self.question_id,
                self.current_docs['system'],
                self.current_docs['user']
            )
        else:
            return template.format(
                self.question_id,
                self.current_docs['user'],
                self.current_docs['system']
            )

    def _launch_new_request(self, timestamp: float, request_executors: List[RequestExecutor]):
        if self.use_sharegpt:
            if self.start_with_gpt:
                prompt = self.sharegpt_data["conversations"][2 * self.question_id + 1][
                    "value"
                ]
            else:
                prompt = self.sharegpt_data["conversations"][2 * self.question_id][
                    "value"
                ]
            self.question_id += 1
        else:
            prompt = self._build_new_question()
        if len(self.chat_history) == 0:
            prompt = self._build_system_prompt() + prompt
        self.chat_history.on_user_query(prompt)
        logger.debug(
            f"User {self.user_config.user_id} issues request {self.question_id}"
        )
        if self.use_sharegpt:
            if self.start_with_gpt:
                max_tokens = self.sharegpt_data["conversations"][2 * self.question_id][
                    "num_tokens"
                ]
            else:
                max_tokens = self.sharegpt_data["conversations"][
                    2 * self.question_id - 1
                ]["num_tokens"]
            max_tokens = min(max_tokens, self.user_config.answer_len)
        else:
            max_tokens = self.user_config.answer_len

        # Launch request to all executors
        for executor in request_executors:
            self.has_unfinished_request[executor.endpoint_name] = True
            executor.launch_request(
                self.chat_history,
                max_tokens,
                self._on_request_finished,
                extra_headers={"x-user-id": str(self.user_config.user_id)},
            )

        self.last_request_time = timestamp

    def _on_request_finished(self, response: Response):
        endpoint = response.endpoint_name
        self.has_unfinished_request[endpoint] = False

        # Only save the conversation from one of the endpoints (Production Stack)
        # to avoid different responses being mixed in the history
        if endpoint == "ProductionStack":
            self.chat_history.on_system_response(response.body)

        logger.debug(
            f"User {self.user_config.user_id} finished one request for endpoint {endpoint}. "
            f"Prompt tokens: {response.prompt_tokens}, "
            f"generation tokens: {response.generation_tokens}"
        )
        self._update_result(response)

        # Write performance metrics to the user request stats file if provided
        if self.real_time_stats_file:
            try:
                # Get the user request stats file by modifying the real_time_stats_file path
                user_request_stats_file = self.real_time_stats_file.replace("real_time_stats", "user_request")

                # Calculate metrics
                ttft_ms = response.ttft * 1000  # Convert to ms

                # Only calculate TPOT if we have more than 1 token
                if response.generation_tokens > 1:
                    tpot_ms = ((response.generation_time - response.ttft) / (response.generation_tokens - 1)) * 1000
                else:
                    tpot_ms = 0

                # Inter-token latency
                itl_ms = (response.generation_time / response.generation_tokens) * 1000 if response.generation_tokens > 0 else 0

                # Format timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(response.finish_time))

                # Create the stats entry
                stats_entry = (
                    f"[{timestamp}] User {self.user_config.user_id}, Request {self.question_id}, Endpoint {endpoint}\n"
                    f"  Documents: {self.current_docs['system']} and {self.current_docs['user']}\n"
                    f"  Prompt tokens: {response.prompt_tokens}\n"
                    f"  Generation tokens: {response.generation_tokens}\n"
                    f"  TTFT: {ttft_ms:.2f} ms\n"
                    f"  TPOT: {tpot_ms:.2f} ms\n"
                    f"  ITL: {itl_ms:.2f} ms\n"
                    f"  Total generation time: {response.generation_time:.2f} s\n"
                    f"  Generation speed: {response.generation_tokens / response.generation_time:.2f} tokens/s\n"
                    f"---------------------------------------------------\n"
                )

                # Append to the file
                with open(user_request_stats_file, "a") as f:
                    f.write(stats_entry)

            except Exception as e:
                logger.error(f"Failed to write to user request stats file: {e}")
                logger.debug(f"Error details: {str(e)}", exc_info=True)

    def set_internal_state(self, offset: float, timestamp: float):
        """Tell the session is the 'offset' seconds after the start"""
        assert len(self.chat_history) == 0, (
            "Internal state should be set " "before the first request"
        )

        num_passed_questions = int(offset / self.user_config.gap_between_requests) + 1

        passed_time = (num_passed_questions - 1) * self.user_config.gap_between_requests

        self.last_request_time = timestamp - offset + passed_time
        self.question_id = num_passed_questions
        logger.debug(
            f"Set internal state for user {self.user_config.user_id}, "
            f"question_id: {self.question_id}, "
            f"last_request_time: {self.last_request_time}"
        )

    def step(self, timestamp: float, request_executors: List[RequestExecutor]):
        if (
            self.question_id >= self.user_config.num_rounds
            and all(not has_unfinished for has_unfinished in self.has_unfinished_request.values())
        ):
            self.finished = True
            return

        if self.last_request_time is None:
            self._launch_new_request(timestamp, request_executors)
            return

        if timestamp - self.last_request_time > self.user_config.gap_between_requests:
            if any(self.has_unfinished_request.values()):
                if timestamp - self.last_unfinished_log > 10:
                    logger.warning(
                        f"User {self.user_config.user_id} has an unfinished "
                        "request and unable to fit the QPS requirement."
                    )
                    self.last_unfinished_log = timestamp
                return

            self._launch_new_request(timestamp, request_executors)
            return

    def summary(self) -> Dict[str, pd.DataFrame]:
        result = {}
        for endpoint, metrics in self.endpoints.items():
            df = pd.DataFrame()
            df["prompt_tokens"] = metrics["prompt_lengths"]
            df["generation_tokens"] = metrics["generation_lengths"]
            df["ttft"] = metrics["ttfts"]
            df["generation_time"] = metrics["generation_times"]
            df["user_id"] = self.user_config.user_id
            df["question_id"] = range(1, len(metrics["prompt_lengths"]) + 1)
            df["launch_time"] = metrics["launch_times"]
            df["finish_time"] = metrics["finish_times"]
            df["endpoint"] = endpoint
            result[endpoint] = df
        return result


class UserSessionManager:

    def __init__(
        self, workload_config: WorkloadConfig, init_user_id=0, use_sharegpt=False, real_time_stats_file=None
    ):
        self.workload_config = workload_config
        self.sessions = []

        gap_between_requests_per_user = workload_config.num_users / workload_config.qps
        session_alive_time = gap_between_requests_per_user * (
            workload_config.num_rounds - 1
        )
        self.gap_between_users = session_alive_time / (workload_config.num_users + 0)
        self.ramp_up_time = workload_config.num_users * self.gap_between_users

        logger.info(
            f"Gap between users: {self.gap_between_users} secs.\n"
            f"Gap between user reqs: {gap_between_requests_per_user} secs.\n"
            f"Expected length of user session: {session_alive_time} secs."
        )

        self.user_id = init_user_id
        self.last_user_join = 0
        self.session_summaries = {}  # Mapping from endpoint to list of session summaries
        self.start_time = None

        self.need_ramp_up = True

        self.use_sharegpt = use_sharegpt
        if self.use_sharegpt:
            self._load_sharegpt_data()

        self.real_time_stats_file = real_time_stats_file

    def _load_sharegpt_data(self):
        with open("ShareGPT.json", "r", encoding="utf-8") as file:
            self.sharegpt_data = json.load(file)
        self.sharegpt_data = [
            d
            for d in self.sharegpt_data
            if d["num_round"] > 2 * self.workload_config.num_rounds
        ]
        logger.info(f"There are {len(self.sharegpt_data)} users satisfying ")

    def _ramp_up(self, timestamp: float, ramp_up_time: float):
        for i in range(self.workload_config.num_users):
            new_session = self._create_user_session()
            offset = ramp_up_time - i * self.gap_between_users
            if offset < 0:
                break
            new_session.set_internal_state(offset, timestamp)
        self.need_ramp_up = False

    def _create_user_session(self):
        self.user_id += 1
        user_config = UserConfig.new_user_config(self.user_id, self.workload_config)
        if self.use_sharegpt:
            user_session = UserSession(
                user_config, self.use_sharegpt, self.sharegpt_data[self.user_id], real_time_stats_file=self.real_time_stats_file
            )
        else:
            user_session = UserSession(user_config, self.use_sharegpt, real_time_stats_file=self.real_time_stats_file)
        self.sessions.append(user_session)
        return user_session

    def _remove_finished_sessions(self):
        sessions_to_remove = [s for s in self.sessions if s.finished]
        if len(sessions_to_remove) > 0:
            logger.info(
                f"Removing {len(sessions_to_remove)} finished sessions, now "
                f"active users: {len(self.sessions) - len(sessions_to_remove)}"
            )
            for session in sessions_to_remove:
                summary_dict = session.summary()
                for endpoint, summary_df in summary_dict.items():
                    if endpoint not in self.session_summaries:
                        self.session_summaries[endpoint] = []
                    self.session_summaries[endpoint].append(summary_df)
        self.sessions = [s for s in self.sessions if not s.finished]

    def step(self, timestamp: float, executors: List[RequestExecutor]):
        if self.need_ramp_up:
            self._ramp_up(timestamp, self.ramp_up_time)

        if self.start_time is None:
            self.start_time = timestamp
            self.last_stats_update = timestamp

        # Update real-time stats every 5 seconds
        if hasattr(self, 'real_time_stats_file') and self.real_time_stats_file and timestamp - getattr(self, 'last_stats_update', 0) > 5:
            self.generate_real_time_stats(timestamp)
            self.last_stats_update = timestamp

        if timestamp - self.last_user_join > self.gap_between_users:
            new_session = self._create_user_session()
            if new_session is not None:
                self.last_user_join = timestamp
                logger.info(
                    f"Joined a new user {self.user_id}, "
                    f"now active users: {len(self.sessions)}"
                )

        for session in self.sessions:
            session.step(timestamp, executors)

        self._remove_finished_sessions()

    def generate_real_time_stats(self, timestamp: float):
        """Generate a real-time stats summary and update the stats file."""
        try:
            # Create temporary summaries for current state
            temp_summaries = {}

            # Collect summaries from finished and active sessions
            for endpoint in set([endpoint for session in self.sessions for endpoint in session.endpoints.keys()] +
                             list(self.session_summaries.keys())):
                # Collect all summaries for this endpoint
                endpoint_summaries = []
                if endpoint in self.session_summaries:
                    endpoint_summaries.extend(self.session_summaries[endpoint])

                # Add summaries from active sessions
                for session in self.sessions:
                    summaries = session.summary()
                    if endpoint in summaries:
                        endpoint_summaries.append(summaries[endpoint])

                if not endpoint_summaries:
                    continue

                df = pd.concat(endpoint_summaries)
                temp_summaries[endpoint] = df

            # If we have data for both endpoints, generate comparison stats
            if "ProductionStack" in temp_summaries and "Other" in temp_summaries:
                ps_df = temp_summaries["ProductionStack"]
                other_df = temp_summaries["Other"]

                # Calculate statistics for both endpoints
                ps_stats = self._calculate_endpoint_stats(ps_df)
                other_stats = self._calculate_endpoint_stats(other_df)

                # Calculate percentage differences (Other relative to ProductionStack)
                # For TTFT, TPOT, ITL - lower is better, so (Other/PS - 1) * 100
                # For throughput - higher is better, so (1 - Other/PS) * 100
                ttft_diff_pct = ((other_stats['ttft_ms'] / ps_stats['ttft_ms'] - 1) * 100) if ps_stats['ttft_ms'] > 0 else 0
                tpot_diff_pct = ((other_stats['tpot_ms'] / ps_stats['tpot_ms'] - 1) * 100) if ps_stats['tpot_ms'] > 0 else 0
                itl_diff_pct = ((other_stats['itl_ms'] / ps_stats['itl_ms'] - 1) * 100) if ps_stats['itl_ms'] > 0 else 0
                throughput_diff_pct = ((1 - other_stats['output_throughput'] / ps_stats['output_throughput']) * 100) if ps_stats['output_throughput'] > 0 else 0

                # Create the stats update
                stats_update = (
                    f"\n=== REAL-TIME STATS UPDATE AT {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    f"Active users: {len(self.sessions)}\n"
                    f"Total requests so far: PS={len(ps_df)}, Other={len(other_df)}\n\n"
                    f"TTFT: ProductionStack={ps_stats['ttft_ms']:.2f}ms, Other={other_stats['ttft_ms']:.2f}ms (Other is {ttft_diff_pct:.1f}% slower)\n"
                    f"TPOT: ProductionStack={ps_stats['tpot_ms']:.2f}ms, Other={other_stats['tpot_ms']:.2f}ms (Other is {tpot_diff_pct:.1f}% slower)\n"
                    f"ITL: ProductionStack={ps_stats['itl_ms']:.2f}ms, Other={other_stats['itl_ms']:.2f}ms (Other is {itl_diff_pct:.1f}% slower)\n"
                    f"Output throughput: ProductionStack={ps_stats['output_throughput']:.2f} t/s, Other={other_stats['output_throughput']:.2f} t/s (Other processed {throughput_diff_pct:.1f}% fewer tokens)\n"
                    f"================================================\n\n"
                )

                # Write the update to the real-time stats file
                with open(self.real_time_stats_file, "a") as f:
                    f.write(stats_update)

                logger.info(f"Updated real-time stats at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            logger.error(f"Error generating real-time stats: {e}")
            logger.debug(f"Error details: {str(e)}", exc_info=True)

    def _calculate_endpoint_stats(self, df):
        """Calculate key statistics for an endpoint."""
        if df.empty:
            return {
                'ttft_ms': 0,
                'tpot_ms': 0,
                'itl_ms': 0,
                'output_throughput': 0
            }

        # TTFT (ms)
        ttft_ms = df["ttft"].mean() * 1000

        # TPOT calculation (excluding first token)
        df['tpot'] = ((df['generation_time'] - df['ttft']) / (df['generation_tokens'] - 1)) * 1000
        tpot = df['tpot'].replace([float('inf'), -float('inf'), np.nan], np.nan).dropna()
        tpot_ms = tpot.mean()

        # ITL calculation
        df['itl'] = (df['generation_time'] / df['generation_tokens']) * 1000
        itl = df['itl'].replace([float('inf'), -float('inf'), np.nan], np.nan).dropna()
        itl_ms = itl.mean()

        # Output throughput
        if df["finish_time"].max() > df["launch_time"].min():
            output_throughput = df["generation_tokens"].sum() / (df["finish_time"].max() - df["launch_time"].min())
        else:
            output_throughput = 0

        return {
            'ttft_ms': ttft_ms,
            'tpot_ms': tpot_ms,
            'itl_ms': itl_ms,
            'output_throughput': output_throughput
        }

    @staticmethod
    def ProcessSummary(
        df: pd.DataFrame,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        pending_queries: int = 0,
        qps: Optional[int] = None,
        output_to_file: bool = False,
        endpoint: str = "default",
        output_file: Optional[str] = None
    ):
        if start_time and end_time:
            launched_queries = len(
                df.query(f"{start_time} <= launch_time <= {end_time}")
            )
            df = df.query(f"{start_time} <= finish_time <= {end_time}")
        else:
            launched_queries = len(df)

        logger.debug(
            f"Endpoint {endpoint} - Launched queries: {launched_queries}, "
            f"pending queries: {pending_queries}, "
            f"finished queries: {len(df)}"
        )

        if qps is None:
            qps = 0.0

        if start_time is None:
            start_time = df["launch_time"].min()
        if end_time is None:
            end_time = df["finish_time"].max()
        total_time = end_time - start_time

        total_requests = launched_queries + pending_queries
        finished_requests = len(df)
        request_throughput = finished_requests / total_time

        total_prompt_tokens = df["prompt_tokens"].sum()
        total_generation_tokens = df["generation_tokens"].sum()
        output_token_throughput = total_generation_tokens / total_time
        total_token_throughput = (total_prompt_tokens + total_generation_tokens) / total_time

        # TTFT stats (in milliseconds)
        ttft_ms = df["ttft"] * 1000
        mean_ttft = ttft_ms.mean()
        median_ttft = ttft_ms.median()
        p99_ttft = np.percentile(ttft_ms, 99)

        # Time per Output Token calculation (excluding first token)
        df['tpot'] = ((df['generation_time'] - df['ttft']) / (df['generation_tokens'] - 1)) * 1000
        tpot = df['tpot'].replace([float('inf'), -float('inf'), np.nan], np.nan).dropna()
        mean_tpot = tpot.mean()
        median_tpot = tpot.median()
        p99_tpot = np.percentile(tpot, 99)

        # Inter-token Latency
        df['itl'] = (df['generation_time'] / df['generation_tokens']) * 1000
        itl = df['itl'].replace([float('inf'), -float('inf'), np.nan], np.nan).dropna()
        mean_itl = itl.mean()
        median_itl = itl.median()
        p99_itl = np.percentile(itl, 99)

        logger.info(f"Calculating performance summary for endpoint {endpoint}")

        # Build the performance summary string
        summary_text = "\n"
        summary_text += f"=========== Endpoint: {endpoint} Benchmark Result ===========\n"
        summary_text += f"Successful requests:                     {finished_requests:<10}\n"
        summary_text += f"Benchmark duration (s):                  {total_time:.2f}      \n"
        summary_text += f"Total input tokens:                      {total_prompt_tokens:<10}\n"
        summary_text += f"Total generated tokens:                  {total_generation_tokens:<10}\n"
        summary_text += f"Request throughput (req/s):              {request_throughput:.2f}      \n"
        summary_text += f"Output token throughput (tok/s):         {output_token_throughput:.2f}    \n"
        summary_text += f"Total Token throughput (tok/s):          {total_token_throughput:.2f}    \n"
        summary_text += "---------------Time to First Token----------------\n"
        summary_text += f"Mean TTFT (ms):                          {mean_ttft:.2f}     \n"
        summary_text += f"Median TTFT (ms):                        {median_ttft:.2f}     \n"
        summary_text += f"P99 TTFT (ms):                           {p99_ttft:.2f}     \n"
        summary_text += "-----Time per Output Token (excl. 1st token)------\n"
        summary_text += f"Mean TPOT (ms):                          {mean_tpot:.2f}     \n"
        summary_text += f"Median TPOT (ms):                        {median_tpot:.2f}     \n"
        summary_text += f"P99 TPOT (ms):                           {p99_tpot:.2f}     \n"
        summary_text += "---------------Inter-token Latency----------------\n"
        summary_text += f"Mean ITL (ms):                           {mean_itl:.2f}     \n"
        summary_text += f"Median ITL (ms):                         {median_itl:.2f}     \n"
        summary_text += f"P99 ITL (ms):                            {p99_itl:.2f}     \n"

        # If not outputting to file, print to console
        if not output_to_file:
            print(f"\n=========== Endpoint: {endpoint} Benchmark Result ===========")
            print(f"Successful requests:                     {finished_requests:<10}")
            print(f"Benchmark duration (s):                  {total_time:.2f}      ")
            print(f"Total input tokens:                      {total_prompt_tokens:<10}")
            print(f"Total generated tokens:                  {total_generation_tokens:<10}")
            print(f"Request throughput (req/s):              {request_throughput:.2f}      ")
            print(f"Output token throughput (tok/s):         {output_token_throughput:.2f}    ")
            print(f"Total Token throughput (tok/s):          {total_token_throughput:.2f}    ")
            print("---------------Time to First Token----------------")
            print(f"Mean TTFT (ms):                          {mean_ttft:.2f}     ")
            print(f"Median TTFT (ms):                        {median_ttft:.2f}     ")
            print(f"P99 TTFT (ms):                           {p99_ttft:.2f}     ")
            print("-----Time per Output Token (excl. 1st token)------")
            print(f"Mean TPOT (ms):                          {mean_tpot:.2f}     ")
            print(f"Median TPOT (ms):                        {median_tpot:.2f}     ")
            print(f"P99 TPOT (ms):                           {p99_tpot:.2f}     ")
            print("---------------Inter-token Latency----------------")
            print(f"Mean ITL (ms):                           {mean_itl:.2f}     ")
            print(f"Median ITL (ms):                         {median_itl:.2f}     ")
            print(f"P99 ITL (ms):                            {p99_itl:.2f}     ")

        # Write to the specified output file if provided
        if output_file:
            try:
                with open(output_file, "a") as f:
                    f.write(summary_text)
                logger.info(f"Performance summary for {endpoint} appended to {output_file}")
            except Exception as e:
                logger.error(f"Failed to write performance summary to file: {e}")

        return df

    def summary(self, start_time: float, end_time: float, output_to_file: bool = True) -> Dict[str, pd.DataFrame]:
        result = {}

        if not self.session_summaries and not self.sessions:
            return {}

        # Get real-time stats filename for final summary output
        stats_filename = self.real_time_stats_file

        # Define CSV directory
        csv_dir = os.path.join(os.path.dirname(stats_filename), "csvs")
        os.makedirs(csv_dir, exist_ok=True)

        for endpoint in set(list(self.session_summaries.keys()) + [endpoint for session in self.sessions for endpoint in session.endpoints.keys()]):
            # Collect all summaries for this endpoint
            endpoint_summaries = []
            if endpoint in self.session_summaries:
                endpoint_summaries.extend(self.session_summaries[endpoint])

            # Add summaries from active sessions
            for session in self.sessions:
                summaries = session.summary()
                if endpoint in summaries:
                    endpoint_summaries.append(summaries[endpoint])

            if not endpoint_summaries:
                continue

            df = pd.concat(endpoint_summaries)
            pending_queries = sum(1 for s in self.sessions if endpoint in s.has_unfinished_request and s.has_unfinished_request[endpoint])

            start_time_actual = max(self.start_time, start_time) if self.start_time else start_time
            end_time_actual = min(end_time, df["finish_time"].max()) if not df.empty else end_time

            qps = self.workload_config.qps

            df = UserSessionManager.ProcessSummary(
                df, start_time_actual, end_time_actual, pending_queries, qps, output_to_file, endpoint, stats_filename
            )
            result[endpoint] = df

            # Save CSV to the new directory
            qps_str = f"{qps:.2f}" if qps is not None else "unknown"
            csv_filename = f"{endpoint}_{qps_str}.csv"
            csv_path = os.path.join(csv_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {endpoint} summary to {csv_path}")

        return result


def warmup_engine(executor):
    logger.info(f"Warming up the {executor.endpoint_name} engine")
    for i in range(10):
        chat_history = ChatHistory()
        chat_history.on_user_query(
            f"WARMUP: Hi, I'm user {i}. Here are some text: {'hi ' * 100}."
        )
        executor.launch_request(chat_history, 100, lambda x: None)

    # Let's not wait for all to complete - we just want to ensure the model is loaded
    time.sleep(1)


def parse_arguments() -> WorkloadConfig:
    parser = argparse.ArgumentParser(description="Parse benchmark configurations.")

    parser.add_argument(
        "--num-users",
        type=int,
        required=True,
        help="Max number of users in the system concurrently",
    )
    parser.add_argument(
        "--shared-system-prompt",
        type=int,
        required=True,
        help="Length of the shared system prompt (tokens)",
    )
    parser.add_argument(
        "--user-history-prompt",
        type=int,
        required=True,
        help="Length of the user-specific history prompt (tokens)",
    )
    parser.add_argument(
        "--answer-len",
        type=int,
        required=True,
        help="Length of the answer in one round",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        required=True,
        help="Number of rounds in the conversation",
    )
    parser.add_argument("--qps", type=float, required=True, help="Overall QPS")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Base URL of the serving engine endpoint (ignored - we use hardcoded endpoints)",
    )
    parser.add_argument(
        "--time",
        type=int,
        required=False,
        help="The time to run the simulation in seconds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="summary.csv",
        help="The output file name for CSV summary files (saved to ../real_time_statistics/csvs/)",
    )
    parser.add_argument(
        "--init-user-id", type=int, default=0, help="The initial user id to start with"
    )
    parser.add_argument(
        "--request-with-user-id",
        action="store_true",
        default=False,
        help="Whether to include user id in request headers",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=30,
        help="The time between two summary loggings in seconds",
    )
    parser.add_argument(
        "--sharegpt", action="store_true", help="Whether to use ShareGPT dataset"
    )
    args = parser.parse_args()
    return args


def parse_process_summary():
    parser = argparse.ArgumentParser(
        description="Parse benchmark configurations.", add_help=False
    )

    parser.add_argument("--process-summary", type=str, default=None)

    args, _ = parser.parse_known_args()
    return args


def process_output(filename):
    logger.warning(
        f"Processing the existing summary file {filename}"
        ", ignoring all the other arguments"
    )
    # Check if the file is in the new directory structure
    if not os.path.exists(filename):
        # Try to find it in the CSV directory
        csv_dir = os.path.join("../real_time_statistics", "csvs")
        potential_path = os.path.join(csv_dir, filename)
        if os.path.exists(potential_path):
            filename = potential_path
        else:
            logger.error(f"Could not find file {filename} or {potential_path}")
            return

    UserSessionManager.ProcessSummary(pd.read_csv(filename), pending_queries=0)


def main():
    args = parse_process_summary()
    if args.process_summary:
        process_output(args.process_summary)
        return

    args = parse_arguments()
    step_interval = 0.1

    # Create output directories
    stats_dir = "../real_time_statistics"
    csv_dir = os.path.join(stats_dir, "csvs")

    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Create file paths for statistics
    real_time_stats_file = os.path.join(stats_dir, f"real_time_stats_{args.qps:.2f}.txt")
    user_request_stats_file = os.path.join(stats_dir, f"user_request_{args.qps:.2f}.txt")

    # Check if contexts directory exists
    if not os.path.exists(CONTEXT_DB_PATH):
        logger.error(f"Contexts directory not found at {CONTEXT_DB_PATH}. Please create it and add document files.")
        return

    # Check for required libraries based on file types
    has_pdf_files = any(f.endswith('.pdf') for f in CONTEXT_FILES)
    has_html_files = any(f.endswith('.html') for f in CONTEXT_FILES)

    if has_pdf_files:
        try:
            import PyPDF2
            logger.info("PyPDF2 library found. PDF processing is available.")
        except ImportError:
            logger.warning("PyPDF2 library not found. PDF files will not be processed correctly.")
            logger.warning("Install PyPDF2 with: pip install PyPDF2")

    if has_html_files:
        try:
            from bs4 import BeautifulSoup
            logger.info("BeautifulSoup library found. HTML processing is available.")
        except ImportError:
            logger.warning("BeautifulSoup library not found. HTML files will not be processed correctly.")
            logger.warning("Install BeautifulSoup with: pip install beautifulsoup4")

    # Check if we have any files
    if not CONTEXT_FILES:
        logger.error(f"No files found in {CONTEXT_DB_PATH}. Please add .txt, .pdf, or .html files to the contexts directory.")
        return

    # Preload all context files
    logger.info(f"Preloading {len(CONTEXT_FILES)} documents from contexts directory...")
    for context_file in CONTEXT_FILES:
        load_context_file(context_file)

    # Clear and initialize the stats file
    with open(real_time_stats_file, 'w') as f:
        f.write(f"Real-time comparison statistics for QPS: {args.qps}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=======================================================\n")
        f.write("Comparing Production Stack (localhost:30080) vs. Other (localhost:30081)\n")
        f.write(f"Using real documents from contexts directory as context\n")
        f.write("Available documents: " + ", ".join(CONTEXT_FILES) + "\n\n")

    # Initialize user request stats file
    with open(user_request_stats_file, 'w') as f:
        f.write(f"Per-user, per-request statistics for QPS: {args.qps}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=======================================================\n\n")

    logger.info(f"Real-time statistics will be written to {real_time_stats_file}")
    logger.info(f"User request statistics will be written to {user_request_stats_file}")

    # Create executors for both endpoints
    executors = [
        RequestExecutor(
            base_url="http://localhost:30080/v1",
            model=args.model,
            endpoint_name="ProductionStack"
        ),
        RequestExecutor(
            base_url="http://localhost:30081/v1",
            model=args.model,
            endpoint_name="Other"
        )
    ]

    # Warm up both executors
    logger.info("Warming up both endpoints")
    for executor in executors:
        warmup_engine(executor)

    workload_config = WorkloadConfig(
        num_users=args.num_users,
        system_prompt_len=args.shared_system_prompt,
        user_info_len=args.user_history_prompt,
        answer_len=args.answer_len,
        num_rounds=args.num_rounds,
        qps=args.qps,
        model=args.model,
        enable_user_id=args.request_with_user_id,
    )

    manager = UserSessionManager(
        workload_config, init_user_id=args.init_user_id, use_sharegpt=args.sharegpt,
        real_time_stats_file=real_time_stats_file
    )

    num_steps = 0
    start_time = time.time()
    last_summary_time = start_time
    try:
        while True:
            num_steps += 1
            manager.step(time.time(), executors)
            time.sleep(step_interval)

            if time.time() - last_summary_time > args.log_interval:
                # Generate intermediate performance summary
                manager.summary(last_summary_time, time.time(), output_to_file=True)
                last_summary_time = time.time()

            if args.time is not None and time.time() - start_time > args.time:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted, waiting for the final result")

    AsyncLoopWrapper.StopLoop()

    logger.info(f"Finished benchmarking, dumping summary to {args.output}")
    summaries = manager.summary(0, time.time(), output_to_file=True)

    # Save individual CSV files for each endpoint
    for endpoint, df in summaries.items():
        # Create CSV filename using args.output base name, put in the csv directory
        csv_filename = f"{endpoint}_{args.output}"
        output_file = os.path.join(csv_dir, csv_filename)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {endpoint} summary to {output_file}")

    # Add final comparison summary to the real-time stats file
    try:
        with open(real_time_stats_file, "a") as f:
            f.write("\n\n=======================================================\n")
            f.write(f"FINAL COMPARISON SUMMARY (QPS: {args.qps})\n")
            f.write("=======================================================\n\n")

            # Compare TTFT and ITL metrics if we have both endpoints
            if "ProductionStack" in summaries and "Other" in summaries:
                ps_df = summaries["ProductionStack"]
                other_df = summaries["Other"]

                ps_ttft_ms = ps_df["ttft"].mean() * 1000
                other_ttft_ms = other_df["ttft"].mean() * 1000

                ps_df['itl'] = (ps_df['generation_time'] / ps_df['generation_tokens']) * 1000
                other_df['itl'] = (other_df['generation_time'] / other_df['generation_tokens']) * 1000

                ps_itl_ms = ps_df['itl'].replace([float('inf'), -float('inf'), np.nan], np.nan).dropna().mean()
                other_itl_ms = other_df['itl'].replace([float('inf'), -float('inf'), np.nan], np.nan).dropna().mean()

                # Add TPOT and throughput to the statistics
                ps_df['tpot'] = ((ps_df['generation_time'] - ps_df['ttft']) / (ps_df['generation_tokens'] - 1)) * 1000
                other_df['tpot'] = ((other_df['generation_time'] - other_df['ttft']) / (other_df['generation_tokens'] - 1)) * 1000

                ps_tpot_ms = ps_df['tpot'].replace([float('inf'), -float('inf'), np.nan], np.nan).dropna().mean()
                other_tpot_ms = other_df['tpot'].replace([float('inf'), -float('inf'), np.nan], np.nan).dropna().mean()

                # Calculate throughput
                ps_throughput = ps_df["generation_tokens"].sum() / (ps_df["finish_time"].max() - ps_df["launch_time"].min())
                other_throughput = other_df["generation_tokens"].sum() / (other_df["finish_time"].max() - other_df["launch_time"].min())

                # Calculate differences to show how much worse Other is
                # For latency metrics (TTFT, TPOT, ITL), lower is better - calculate how much slower Other is
                ttft_diff = other_ttft_ms - ps_ttft_ms
                ttft_pct = ((other_ttft_ms / ps_ttft_ms) - 1) * 100 if ps_ttft_ms > 0 else 0

                tpot_diff = other_tpot_ms - ps_tpot_ms
                tpot_pct = ((other_tpot_ms / ps_tpot_ms) - 1) * 100 if ps_tpot_ms > 0 else 0

                itl_diff = other_itl_ms - ps_itl_ms
                itl_pct = ((other_itl_ms / ps_itl_ms) - 1) * 100 if ps_itl_ms > 0 else 0

                # For throughput, higher is better - calculate how much slower Other is
                throughput_diff = ps_throughput - other_throughput
                throughput_pct = (1 - (other_throughput / ps_throughput)) * 100 if ps_throughput > 0 else 0

                f.write("Key Performance Comparison:\n")
                f.write(f"TTFT: ProductionStack={ps_ttft_ms:.2f}ms, Other={other_ttft_ms:.2f}ms (Other is {ttft_pct:.1f}% slower)\n")
                f.write(f"TPOT: ProductionStack={ps_tpot_ms:.2f}ms, Other={other_tpot_ms:.2f}ms (Other is {tpot_pct:.1f}% slower)\n")
                f.write(f"ITL: ProductionStack={ps_itl_ms:.2f}ms, Other={other_itl_ms:.2f}ms (Other is {itl_pct:.1f}% slower)\n")
                f.write(f"Throughput: ProductionStack={ps_throughput:.2f}t/s, Other={other_throughput:.2f}t/s (Other processed {throughput_pct:.1f}% fewer tokens)\n\n")

                # Add document usage statistics
                f.write("Document Usage Statistics:\n")
                doc_counts = {doc: 0 for doc in CONTEXT_FILES}
                total_sessions = 0

                for session in manager.sessions:
                    if session.current_docs["system"] and session.current_docs["user"]:
                        total_sessions += 1
                        doc_counts[session.current_docs["system"]] += 1
                        doc_counts[session.current_docs["user"]] += 1

                f.write(f"Total unique sessions with documents: {total_sessions}\n")
                f.write("Document frequency:\n")

                for doc, count in doc_counts.items():
                    if count > 0:
                        f.write(f"  {doc}: {count} times ({count/(total_sessions*2)*100:.1f}%)\n")

            f.write("\nTest completed at: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

    except Exception as e:
        logger.error(f"Failed to write final comparison summary: {e}")
        logger.exception(e)


if __name__ == "__main__":
    main()