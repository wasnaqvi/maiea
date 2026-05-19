import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / '.env')  # Load environment variables from repo-local .env file
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / 'workspace' / '.matplotlib'))


def require_openai_credentials():
    if os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_ADMIN_KEY'):
        return

    raise RuntimeError(
        'Missing OpenAI credentials. Create a .env file next to run_aster.py '
        'with OPENAI_API_KEY=your_api_key, or export OPENAI_API_KEY before '
        'running python run_aster.py.'
    )


def require_groq_credentials():
    if os.getenv('GROQ_API_KEY'):
        return

    raise RuntimeError(
        'Missing Groq credentials. Create a .env file next to run_aster.py '
        'with GROQ_API_KEY=your_api_key, then set ASTER_LLM_PROVIDER=groq '
        'before running python run_aster.py.'
    )


def get_llm_provider():
    return os.getenv('ASTER_LLM_PROVIDER', 'ollama').lower()

from orchestral import Agent
from orchestral.tools import (
    RunCommandTool,
    WriteFileTool,
    ReadFileTool,
    EditFileTool,
    FileSearchTool,
    WebSearchTool as OrchestralWebSearchTool,
    TodoWrite,
    TodoRead,
    DisplayImageTool
)
from orchestral.tools.hooks import DangerousCommandHook
from orchestral.prompts import RICH_UI_SYSTEM_PROMPT
# from orchestral.llm import Claude
from orchestral.llm import GPT, Ollama
from openai import OpenAI


class WebSearchTool(OrchestralWebSearchTool):
    """Web search tool that can be deep-copied by Orchestral Agent setup."""

    def __deepcopy__(self, memo=None):
        return type(self)(search_context_size=self.search_context_size)


class GroqGPT(GPT):
    """OpenAI-compatible Groq client with retries for malformed tool calls."""

    def call_api(self, formatted_input, use_prompt_cache=False, **kwargs):
        call_params = {
            "model": self.model,
            "messages": formatted_input,
            "temperature": float(os.getenv("GROQ_TEMPERATURE", "0.2")),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            **kwargs,
        }

        if self.tool_schemas:
            call_params["tools"] = self.tool_schemas

        max_retries = int(os.getenv("GROQ_TOOL_RETRIES", "3"))
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(**call_params)
            except Exception as exc:
                if not self._is_failed_tool_generation(exc) or attempt == max_retries - 1:
                    raise
                call_params["temperature"] = max(call_params["temperature"] - 0.1, 0.0)

        return self.client.chat.completions.create(**call_params)

    @staticmethod
    def _is_failed_tool_generation(exc: Exception) -> bool:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict) and error.get("failed_generation"):
                return True
        return "failed_generation" in str(exc)


def build_llm(provider: str):
    model = os.getenv('ASTER_LLM_MODEL')

    if provider == 'ollama':
        try:
            return Ollama(
                model=model or 'llama3.2:latest',
                host=os.getenv('OLLAMA_HOST') or None
            )
        except Exception as exc:
            raise SystemExit(
                'Could not connect to Ollama. Install/start Ollama, pull a model, '
                'then run again. Example: ollama pull llama3.2. If Ollama is '
                'running on a non-default host, set OLLAMA_HOST in .env.'
            ) from None

    if provider == 'openai':
        require_openai_credentials()
        return GPT(model=model or 'gpt-4.1-mini')

    if provider == 'groq':
        require_groq_credentials()
        llm = GroqGPT(model=model or 'openai/gpt-oss-120b', api_key=os.getenv('GROQ_API_KEY'))
        llm.client = OpenAI(
            api_key=os.getenv('GROQ_API_KEY'),
            base_url=os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1'),
            timeout=30.0,
        )
        return llm

    raise ValueError(
        f"Unsupported ASTER_LLM_PROVIDER='{provider}'. "
        "Supported providers: ollama, openai, groq."
    )

from aster_toolkit import (
    RunTaurexModelTool,
    SetTaurexPaths,
    SimulateTaurexRetrieval,
    PlotCornerPosteriors,
    GetExoplanetParameters,
    DownloadDataset,
    FindExoplanetsByCondition,
    SearchMastJwstObservations,
    GetMastObservationProducts,
    DownloadMastJwstProducts,
)

base_directory = 'workspace'
os.makedirs(base_directory, exist_ok=True)
llm_provider = get_llm_provider()

tools = [
    # File and command tools
    RunCommandTool(base_directory=base_directory, persistent=True),
    WriteFileTool(base_directory=base_directory),
    ReadFileTool(base_directory=base_directory, show_line_numbers=True),
    EditFileTool(base_directory=base_directory),
    FileSearchTool(base_directory=base_directory),
    TodoRead(),
    TodoWrite(initial_todos='- [ ] Sample todo item'),
    DisplayImageTool,

    # TauREx modeling tools
    SetTaurexPaths,
    RunTaurexModelTool(base_directory=base_directory),
    SimulateTaurexRetrieval(base_directory=base_directory),
    PlotCornerPosteriors(base_directory=base_directory),

    # Data acquisition tools
    GetExoplanetParameters(),
    DownloadDataset(base_directory=base_directory),
    FindExoplanetsByCondition(),

    # MAST / JWST tools
    SearchMastJwstObservations(),
    GetMastObservationProducts(),
    DownloadMastJwstProducts(base_directory=base_directory),
]

if llm_provider == 'openai':
    tools.insert(5, WebSearchTool())

hooks = [DangerousCommandHook()]


def build_agent():
    # Load ASTER system prompt
    with open('aster_system_prompt.md', 'r') as f:
        aster_prompt = f.read()

    system_prompt = f'{RICH_UI_SYSTEM_PROMPT}\n\n{aster_prompt}'
    if llm_provider == 'groq':
        system_prompt += (
            "\n\nGroq tool-use instruction: when you need a tool, emit exactly one "
            "valid function call with JSON arguments that match the tool schema. "
            "Do not write Python dict syntax, comments, or explanatory text inside "
            "the function-call arguments."
        )

    return Agent(
        # llm=Claude(),
        llm=build_llm(llm_provider),
        tools=tools,
        tool_hooks=hooks,
        system_prompt=system_prompt
    )


def main():
    agent = build_agent()

    from orchestral.ui.app import server as app_server

    if llm_provider == 'groq':
        app_server.state.non_streaming_models.add(agent.llm.model)

    app_server.run_server(agent, host='localhost', port=8000)


if __name__ == '__main__':
    main()
