import os
import numpy as np, pandas as pd, re, json, os, shutil, inspect
import nbformat
import contextlib
import io
import os
import re
import subprocess
import traceback
import sys

from io import StringIO
from pathlib import Path
from insightbench import prompts
from copy import deepcopy
from typing import Dict
from insightbench import tools
from dateutil.parser import parse
from langchain.schema import HumanMessage, SystemMessage
from warnings import warn
from functools import partial
from langchain.prompts import PromptTemplate
from openai import OpenAI
from pathlib import Path

# from langchain_community.chat_models import ChatOpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JSON_MAX_TOKENS = 5000
JSON_MAX_CHARS = JSON_MAX_TOKENS * 4  # 4 chars per token (roughly)


def interpret_solution(
    solution: dict,
    n_retries: int,
    model_name: str,
    prompt_method,
    schema,
    temperature: 0,
    seed=None,
) -> str:
    """
    Produce insights for a task based on a solution output by a model

    Parameters:
    -----------
    solution: dict
        The output of the code generation function
    answer_template: dict
        A template for the answer that the human should provide. This template should contain a "results" tag
        that contains a list of expected results in the form of dictionaries. Each dictionary should contain
        the following keys: "name", "description", and "value". The model will be asked to fill in the values.
    model: str
        The name of the model to use (default: gpt-4)
    n_retries: int
        The number of times to retry the interpretation if it fails

    Returns:
    --------
    solution_path: str
        The path to the input solution file, which has been updated with the interpretation

    """
    prompt_template = prompts.get_interpret_prompt(method=prompt_method)
    # create prompt
    prompt = PromptTemplate.from_template(prompt_template)

    insight_prompt = _build_insight_prompt(solution)

    # instantiate llm model
    llm = get_chat_model(model_name, temperature, seed)

    # Get human readable answer
    out, _ = retry_on_parsing_error(
        llm,
        prompt.format(
            goal=solution["goal"],
            question=solution["question"],
            message=solution["code_output"],
            insights=insight_prompt,
            schema=schema,
        ),
        parser=_parse_human_readable_insight,
        n_retries=n_retries,
    )
    solution["interpretation"] = out
    return solution


def extract_python_code_blocks(text):
    """
    Extract and merge Python code blocks from a given text string.

    The function identifies code blocks that start with ``` or ```python and end with ```.
    After extracting the code blocks, it removes the start and end delimiters (```, ```python),
    and merges the code blocks together into a single string.

    Parameters
    ----------
    text : str
        The input string from which Python code blocks need to be extracted.

    Returns
    -------
    str
        A string containing the merged Python code blocks stripped of leading and trailing whitespaces.
        Code blocks are separated by a newline character.

    """

    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return "\n".join(block.strip() for block in code_blocks)

    # Some models emit an opening fence but omit the closing fence.  In that
    # case the old parser returned the whole response, including ```python,
    # and PythonREPL.exec() raised a SyntaxError.  Recover the code payload
    # before handing it to the executor.
    opening = re.search(r"```(?:python)?\s*", text, re.IGNORECASE)
    if opening:
        text = text[opening.end():]
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def normalize_generated_code(code):
    """Add the stable plotting-tool imports expected by the executor.

    Models sometimes call ``plot_countplot`` or ``plot_lines`` while only
    importing ``save_json`` and ``fix_fnames``. That is an executor/API
    compatibility failure rather than a data-analysis failure. The prelude
    keeps the generated code executable and also exposes the module-qualified
    ``tools`` form used by other completions.
    """
    prelude = (
        "import insightbench.tools as tools\n"
        "from insightbench.tools import *\n\n"
    )
    stripped = code.lstrip()
    if "import insightbench.tools as tools" in stripped and "from insightbench.tools import *" in stripped:
        return code
    return prelude + code


class PythonREPL:
    """
    Simulates a standalone Python REPL.

    TODO add a way to pass a random seed to the REPL
    """

    def __init__(self):
        self.history = []

    def run(self, command: str, workdir: str = None) -> str:
        """Run command with own globals/locals and returns anything printed."""

        if workdir is not None:
            old_cwd = Path.cwd()
            os.chdir(workdir)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            try:
                exec(command, locals())
                valid = True
                retry_message = ""
                self.history.append((command, workdir))
            except Exception as e:
                valid = False
                retry_message = traceback.format_exc() + "\n" + str(e)
            finally:
                if workdir is not None:
                    os.chdir(old_cwd)
        output = buffer.getvalue()

        return output, valid, retry_message

    def clone(self):
        """Clone the REPL from history.

        it is not possible to clone the REPL from the globals/locals because they
        may contain references to objects that cannot be pickled e.g. python modules.
        Instead, we clone the REPL by replaying the history.
        """
        new_repl = PythonREPL()
        # deepcopy of history
        new_repl.history = deepcopy(self.history)

        for command, workdir in self.history:
            new_repl.run(command, workdir=workdir)

        return new_repl


def _execute_command(args):
    """Execute a command and return the stdout, stderr and return code

    Parameters
    ----------
    args : list of str or str, directly passed to subprocess.Popen

    Returns
    -------
    stdout : str
        stdout of the command
    stderr : str
        stderr of the command
    returncode : int
        return code of the command
    """
    try:
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False
        )
    except FileNotFoundError as e:
        return "", str(e), 1

    stdout, stderr = process.communicate()

    # decode bytes object to string
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    return stdout, stderr, process.returncode


def pip_install(requirements):
    """Install a list of requirements using pip

    Parameters
    ----------
    requirements : list of str
        List of requirements to install, should be in the format of pip install <requirement>

    Returns
    -------
    stdout : str
        stdout of the command
    valid : bool
        True if the installation was successful, False otherwise
    retry_message : str
        Error message if the installation was not successful
    """

    if isinstance(requirements, str):
        requirements = [requirements]
    retry_messages = []
    stdouts = []
    for req in requirements:
        stdout, stderr, code = _execute_command(["pip", "install", req])
        stdouts.append(stdout)
        if stdout.strip().startswith("Usage:"):
            retry_messages.append(
                f"Seems like there is an error on the pip commandline, it just prints usage. stderr:\n{stderr}. stdout:\n{stdout[-1000:]}"
            )
        if code != 0:
            retry_messages.append(
                f"Error code {code} when installing {req}. stderr:\n{stderr}. stdout:\n{stdout[-1000:]}"
            )

    valid = len(retry_messages) == 0
    retry_message = "\n".join(retry_messages)
    return "\n".join(stdouts), valid, retry_message


# =============================================================================
# Code generation
# =============================================================================
def _code_parser(code, output_folder):
    """
    A parser that is used to parse the code generated by the LLM
    and determine whether it is acceptable or not

    """
    # Clean output folder
    output_folder = Path(output_folder)
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True)

    # Extract code blocks from the input code (might contain other text)
    code_block = extract_python_code_blocks(code)
    if len(code_block) == 0:
        # No code blocks detected so input is likely already raw code
        code_block = code
    code_block = normalize_generated_code(code_block)

    # Run code and report any errors
    output, valid, retry_message = PythonREPL().run(code_block, workdir=output_folder)
    if not valid:
        return "", valid, retry_message

    # Validate output files
    json_files = [f.name for f in output_folder.glob("*.json")]
    plot_files = [f.name for f in output_folder.glob("*.jpg")]

    try:
        # assert that there is x_axis.json, y_axis.json, and stat.json in json_files
        assert "x_axis.json" in json_files
        assert "y_axis.json" in json_files
        assert "stat.json" in json_files
        assert len(json_files) == 3

        # Check that the total length of all json files is not too long
        json_lengths = [len(open(output_folder / f).read()) for f in json_files]
        total_json_chars = sum(json_lengths)
        if total_json_chars > JSON_MAX_CHARS:
            return (
                "",
                False,
                f"Error: The total length of your json files cannot exceed {JSON_MAX_CHARS} characters. Here is the total length of each json file: {', '.join(f'{f} ({l} characters)' for f, l in zip(json_files, json_lengths))}.",
            )

        assert len(plot_files) == 1 and "plot.jpg" in plot_files

    except:
        return (
            "",
            False,
            f"Error: Your code did not generate the expected output files. Expected x_axis.json, y_axis.json, and stat.json files.",
        )

    # All checks have passed!
    return output, True, ""


def retry_on_parsing_error(
    llm,
    initial_prompt,
    parser,
    n_retries,
    exception_on_max_retries=True,
):
    """
    Try querying a LLM until it returns a valid value with a maximum number of retries.

    Parameters:
    -----------
    llm : callable
        A langchain LLM model.
    initial_prompt : str
        The initial prompt to send to the LLM.
    parser : callable
        A function taking a message and returning a tuple (value, valid, retry_message),
        where retries will be made until valid is True.
    n_retries : int
        The maximum number of retries.
    exception_on_max_retries : bool
        If True, raise an exception if the maximum number of retries is reached.
        Otherwise, returns "".

    Returns:
    --------
    value : str
        The value returned by the LLM.
    completions : list
        The attempts made by the LLM.

    """
    retry_template = prompts.RETRY_TEMPLATE
    prompt = initial_prompt

    completions = []
    for i in range(n_retries + 1):  # Add one since the initial prompt is not a retry
        # Try to get a valid completion
        completions.append(llm(prompt))
        output, valid, retry_message = parser(completions[-1])

        # If parser execution succeeds return the output
        if valid:
            return output, completions

        # If parser execution fails, produce a new prompt that includes the previous output and the error message
        warn(
            f"Retry {i+1}/{n_retries} - Query failed with error: {retry_message}",
            RuntimeWarning,
        )
        prompt = retry_template.format(
            initial_prompt=initial_prompt,
            prev_output=completions[-1],
            error=retry_message,
        )

    if exception_on_max_retries:
        return f"Could not parse a valid value after {n_retries} retries.", [
            "```python\nimport pandas as pd```",
            "```python\nimport numpy as np```",
        ]
    else:
        return retry_message, completions


def _extract_top_values(values, k=5, max_str_len=100):
    """
    Extracts the top k values from a pandas series

    Parameters
    ----------
    values : pandas.Series
        Series to extract top values from
    k : int, optional
        Number of top values to extract, by default 5
    max_str_len : int, optional
        Maximum length of string values (will be truncated), by default 100

    """
    top = values.value_counts().iloc[:k].index.values.tolist()
    top = [x if not isinstance(x, str) else x[:max_str_len] for x in top]
    return top


def get_schema(df):
    """
    Extracts schema from a pandas dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to extract schema from

    Returns
    -------
    list of dict
        Schema for each column in the dataframe

    """
    schema = []

    for col in df.columns:
        info = {
            "name": col,
            "type": df[col].dtype,
            "missing_count": df[col].isna().sum(),
            "unique_count": df[col].unique().shape[0],
        }

        # If the column is numeric, extract some stats
        if np.issubdtype(df[col].dtype, np.number):
            info["min"] = df[col].min()
            info["max"] = df[col].max()
            info["mean"] = df[col].mean()
            info["std"] = df[col].std()
        # If the column is a date, extract the min and max
        elif _is_date(df[col].iloc[0]):
            info["min"] = df[col].dropna().min()
            info["max"] = df[col].dropna().max()
        # If the column is something else, extract the top values
        else:
            info["top5_unique_values"] = _extract_top_values(df[col])

        schema.append(info)

    return schema


def schema_to_str(schema) -> str:
    """Converts the list of dict to a promptable string.

    Parameters
    ----------
    schema : list of dict
        Schema for each column in the dataframe

    Returns
    -------
    str
        String representation of the schema
    """
    schema_str = ""
    for col in schema:
        schema_str += f"Column: {col['name']} ({col['type']})\n"
        for key, val in col.items():
            if key in ["name", "type"]:
                continue
            schema_str += f"  {key}: {val}\n"
    return schema_str


def _is_date(string):
    """
    Checks if a string is a date

    Parameters
    ----------
    string : str
        String to check

    Returns
    -------
    bool
        True if the string is a date, False otherwise

    """
    try:
        parse(str(string))
        return True
    except ValueError:
        return False


def schema_to_str(schema) -> str:
    """Converts the list of dict to a promptable string.

    Parameters
    ----------
    schema : list of dict
        Schema for each column in the dataframe

    Returns
    -------
    str
        String representation of the schema
    """
    schema_str = ""
    for col in schema:
        schema_str += f"Column: {col['name']} ({col['type']})\n"
        for key, val in col.items():
            if key in ["name", "type"]:
                continue
            schema_str += f"  {key}: {val}\n"
    return schema_str


def convert_messages_to_text(messages):
    """
    Convert a list of messages to a string

    Parameters
    ----------
    messages : list
        List of messages to convert

    Returns
    -------
    str
        String representation of the messages

    """
    return "\n".join(
        [
            (
                f"[INST]\n{m.content}\n[/INST]"
                if m.type in ["system", "agent"]
                else f"\n{m.content}\n"
            )
            for m in messages
        ]
    )


def chat_and_retry(chat, messages, n_retry, parser):
    """
    Retry querying the chat models until it returns a valid value with a maximum number of retries.

    Parameters:
    -----------
        chat: callable
            A langchain chat object taking a list of messages and returning the llm's message.
        messages: list
            The list of messages so far.
        n_retry: int
            The maximum number of retries.
        parser: callable
            A function taking a message and returning a tuple (value, valid, retry_message)
            where value is the parsed value, valid is a boolean indicating if the value is valid and retry_message
            is a message to display to the user if the value is not valid.

    Returns:
    --------
        value: object
            The parsed value.

    Raises:
    -------
        ValueError: if the value could not be parsed after n_retry retries.

    """
    # Convert the structured message objects only once. Previously `messages`
    # was overwritten with a string inside the loop; after an invalid model
    # response, the next retry passed that string back to
    # `convert_messages_to_text`, which iterated over characters and raised
    # `AttributeError: 'str' object has no attribute 'type'`.
    prompt_text = (
        messages if isinstance(messages, str) else convert_messages_to_text(messages)
    )
    for i in range(n_retry):
        answer = chat(prompt_text)
        value, valid, retry_message = parser(answer)

        if valid:
            return value

        msg = f"Query failed. Retrying {i+1}/{n_retry}.\n[LLM]:\n{answer}\n[User]:\n{retry_message}"
        warn(msg, RuntimeWarning)
        prompt_text += f"\n[LLM]:\n{answer}\n[User]:\n{retry_message}\n"

    return {
        "answer": "Error occured",
        "justification": f"Could not parse a valid value after {n_retry} retries.",
    }


def extract_html_tags(text, keys):
    """Extract the content within HTML tags for a list of keys.

    Parameters
    ----------
    text : str
        The input string containing the HTML tags.
    keys : list of str
        The HTML tags to extract the content from.

    Returns
    -------
    dict
        A dictionary mapping each key to a list of subset in `text` that match the key.

    Notes
    -----
    All text and keys will be converted to lowercase before matching.

    """
    content_dict = {}
    keys = set(keys)
    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        matches = re.findall(pattern, text, re.DOTALL)
        # print(matches)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


def _parse_human_readable_insight(output):
    """
    A parser that makes sure that the human readable insight is produced in the correct format

    """
    try:
        answer = extract_html_tags(output, ["answer"])
        if "answer" not in answer:
            return (
                "",
                False,
                f"Error: you did not generate answers within the <answer></answer> tags",
            )
        answer = answer["answer"][0]
    except ValueError as e:
        return (
            "",
            False,
            f"The following error occured while extracting the value for the <answer> tag: {str(e)}",
        )

    try:
        justification = extract_html_tags(output, ["justification"])
        if "justification" not in justification:
            return (
                "",
                False,
                f"Error: you did not generate answers within the <justification></justification> tags",
            )
        justification = justification["justification"][0]
    except ValueError as e:
        return (
            "",
            False,
            f"The following error occured while extracting the value for the <justification> tag: {str(e)}",
        )
    try:
        insight = extract_html_tags(output, ["insight"])
        if "insight" not in insight:
            return (
                "",
                False,
                f"Error: you did not generate answers within the <insight></insight> tags",
            )
        insight = insight["insight"][0]
    except ValueError as e:
        return (
            "",
            False,
            f"The following error occured while extracting the value for the <insight> tag: {str(e)}",
        )

    return (
        {"answer": answer, "justification": justification, "insight": insight},
        True,
        "",
    )


def _build_insight_prompt(solution) -> str:
    """
    Gather all plots and statistics produced by the model and format then nicely into text

    """
    insight_prompt = ""
    for i, var in enumerate(solution["vars"]):
        insight_prompt += f"<insight id='{i}'>"
        insight_prompt += f"    <stat>"
        insight_prompt += f"        <name>{var['stat'].get('name', 'n/a')}</name>"
        insight_prompt += f"        <description>{var['stat'].get('description', 'n/a')}</description>"
        stat_val = var["stat"].get("value", "n/a")
        stat_val = stat_val[:50] if isinstance(stat_val, list) else stat_val
        insight_prompt += f"        <value>{stat_val}</value>"
        insight_prompt += f"    </stat>"
        insight_prompt += f"    <plot filename='{var['plot']['name']}'>"
        insight_prompt += f"        <xaxis>"
        insight_prompt += f"            <description>{var['x_axis'].get('description', 'n/a')}</description>"
        x_val = var["x_axis"].get("value", "n/a")
        x_val = x_val[:50] if isinstance(x_val, list) else x_val
        insight_prompt += f"            <value>{x_val}</value>"
        insight_prompt += f"        </xaxis>"
        insight_prompt += f"        <yaxis>"
        insight_prompt += f"            <description>{var['y_axis'].get('description', 'n/a')}</description>"
        y_val = var["y_axis"].get("value", "n/a")
        y_val = y_val[:50] if isinstance(y_val, list) else y_val
        insight_prompt += f"            <value>{y_val}</value>"
        insight_prompt += f"        </yaxis>"
        insight_prompt += f"    </plot>"
        insight_prompt += f"</insight>"
    return insight_prompt


def get_insights(
    context,
    goal,
    messages=[],
    schema=None,
    max_questions=3,
    model_name="gpt-4o",
    temperature=0,
):

    chat = get_chat_model(model_name, temperature)

    prompt = prompts.GET_INSIGHTS_TEMPLATE
    messages = [
        SystemMessage(content=prompts.GET_INSIGHTS_SYSTEM_MESSAGE),
        HumanMessage(
            content=prompt.format(
                context=context, goal=goal, schema=schema, max_questions=max_questions
            )
        ),
    ]

    def _validate_tasks(out):
        isights = extract_html_tags(out, ["insight"])

        # Check that there are insights generated
        if "insight" not in isights:
            return (
                out,
                False,
                f"Error: you did not generate insights within the <insight></insight> tags.",
            )
        isights = isights["insight"]
        print("The insights are:", isights)
        print("Length:", len(isights), "   Max:", max_questions)
        return (isights, out), True, ""

    insights, message = chat_and_retry(
        chat, messages, n_retry=3, parser=_validate_tasks
    )

    return insights


def get_questions(
    prompt_method,
    context,
    goal,
    messages=[],
    schema=None,
    max_questions=10,
    model_name="gpt-4o",
    temperature=0,
    seed=None,
):
    if prompt_method is None:
        prompt_method = "basic"

    prompt, system = prompts.get_question_prompt(method=prompt_method)

    chat = get_chat_model(model_name, temperature, seed)

    messages = [
        SystemMessage(content=system),
        HumanMessage(
            content=prompt.format(
                context=context, goal=goal, schema=schema, max_questions=max_questions
            )
        ),
    ]

    def _validate_tasks(out):
        questions = extract_html_tags(out, ["question"])
        if "question" not in questions:
            return (
                out,
                False,
                f"Error: you did not generate questions within the <question></question> tags",
            )
        questions = questions["question"]
        # Check that there are at most max_questions questions
        if len(questions) > max_questions:
            return (
                out,
                False,
                f"Error: you can only ask at most {max_questions} questions, but you asked {len(questions)}.",
            )

        return (questions, out), True, ""

    questions, message = chat_and_retry(
        chat, messages, n_retry=3, parser=_validate_tasks
    )

    return questions


def get_dataset_description(
    prompt,
    system,
    context,
    goal,
    messages=[],
    schema=None,
    model_name="gpt-4o",
    temperature=0,
    seed=None,
):

    chat = get_chat_model(model_name, temperature, seed)

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=prompt.format(context=context, goal=goal, schema=schema)),
    ]

    def _validate_tasks(out):
        try:
            questions = extract_html_tags(out, ["description"])["description"]
        except Exception as e:
            return (
                out,
                False,
                f"Error: {str(e)}",
            )

        return (questions, out), True, ""

    data_description, message = chat_and_retry(
        chat, messages, n_retry=2, parser=_validate_tasks
    )

    return data_description


def get_follow_up_questions(
    context,
    goal,
    question,
    answer,
    schema=None,
    max_questions=3,
    model_name="gpt-4o",
    prompt_method=None,
    question_type="descriptive",
    temperature=0,
    seed=None,
):
    if prompt_method is None:
        prompt_method = "follow_up"

    prompt, system = prompts.get_question_prompt(method=prompt_method)
    chat = get_chat_model(model_name, temperature, seed)

    if prompt_method == "follow_up_with_type":
        content = prompt.format(
            context=context,
            goal=goal,
            question=question,
            answer=answer,
            schema=schema,
            max_questions=max_questions,
            question_type=question_type,
        )

    else:
        content = prompt.format(
            context=context,
            goal=goal,
            question=question,
            answer=answer,
            schema=schema,
            max_questions=max_questions,
        )

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=content),
    ]

    def _validate_tasks(out):
        extracted = extract_html_tags(out, ["question"])
        if "question" not in extracted or not extracted["question"]:
            return (
                out,
                False,
                "Error: you did not generate questions within the <question></question> tags",
            )
        questions = extracted["question"]
        # print("The questions are:", questions)
        # print("Length:", len(questions), "   Max:", max_questions)

        # Check that there are at most max_questions questions
        if len(questions) > max_questions:
            return (
                out,
                False,
                f"Error: you can only ask at most {max_questions} questions, but you asked {len(questions)}.",
            )

        return (questions, out), True, ""

    questions, message = chat_and_retry(
        chat, messages, n_retry=3, parser=_validate_tasks
    )

    return questions


def select_a_question(
    questions,
    context,
    goal,
    prev_questions,
    model_name="gpt-4o",
    prompt_template=None,
    system_template=None,
    temperature=0,
    seed=None,
):

    chat = get_chat_model(model_name, temperature, seed)

    followup_questions_formatted = "\n".join(
        [f"{i+1}. {q}\n" for i, q in enumerate(questions)]
    )
    if prev_questions:
        prev_questions_formatted = "\n".join(
            [f"{i+1}. {q}\n" for i, q in enumerate(prev_questions)]
        )
    else:
        prev_questions_formatted = None

    prompt = prompt_template
    messages = [
        SystemMessage(content=system_template),
        HumanMessage(
            content=prompt.format(
                context=context,
                goal=goal,
                prev_questions_formatted=prev_questions_formatted,
                followup_questions_formatted=followup_questions_formatted,
            )
        ),
    ]

    def _validate_tasks(out):
        extracted = extract_html_tags(out, ["question_id"])
        if "question_id" not in extracted or not extracted["question_id"]:
            return (
                out,
                False,
                "Error: you did not select a question within the <question_id></question_id> tags.",
            )
        question_id = extracted["question_id"][0]
        try:
            question_id = int(question_id)
        except (TypeError, ValueError):
            return (
                out,
                False,
                "Error: question_id must be an integer index.",
            )
        # Check that there are at most max_questions questions
        if question_id < 0 or question_id >= len(questions):
            return (
                out,
                False,
                f"Error: selected question index should be between 0-{len(questions)-1}.",
            )
        return (question_id, out), True, ""

    question_id, message = chat_and_retry(
        chat, messages, n_retry=3, parser=_validate_tasks
    )
    return question_id


def _shared_analysis_tool_context():
    """Expose the local comparison toolkit even when execution changes cwd."""
    repo_root = Path(__file__).resolve().parents[3]
    if (repo_root / "baseline_support" / "analysis_tools.py").is_file():
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    from baseline_support.analysis_tools import SHARED_TOOL_REFERENCE, analysis_tool_metadata

    adapter = """
SHARED TRAJECTORY TOOLKIT FOR INSIGHTBENCH
The following helpers come from baseline_support.analysis_tools, not insightbench.tools.
They are the same computational implementations and documentation used by TrajStoryAgent.
Prefer these helpers for routine trajectory computations; custom pandas/numpy analyses
and existing insightbench.tools remain available. Import the helpers you actually call.
Schema-column restrictions apply to raw source reads: derived columns returned by
documented helpers or your computations may be used after they have been created.

Keep the InsightBench output contract: read the supplied CSV paths with pandas and save
exactly plot.jpg, stat.json, x_axis.json, and y_axis.json. The shared plotting helpers
return figures: call fig.savefig("plot.jpg", bbox_inches="tight"), then plt.close("all").
Save actual plotted values using insightbench.tools.save_json(data_dict, ftype), with
ftype="stat", "x_axis", or "y_axis", and call tools.fix_fnames() as usual.
statistic()/table_from_frame() return in-memory dictionaries, not saved files.
AnalysisSession is TrajStoryAgent's native artifact writer: do not call its emit_result
or produce raw_analysis_result.json in InsightBench. Use the computational helpers below
with InsightBench's own output writer. No additional analysis or review stage is required.
""".strip()
    return adapter + "\n\n" + SHARED_TOOL_REFERENCE, analysis_tool_metadata()


def generate_code(
    schema,
    user_schema,
    goal,
    question,
    database_path,
    user_database_path,
    output_folder,
    n_retries,
    prompt_method=None,
    model_name="gpt-4o",
    temperature=0,
    table_schemas=None,
    database_paths=None,
    seed=None,
    shared_analysis_tools=False,
):
    """
    Solve a task using the naive single step approach

    See main function docstring for more details

    """
    if table_schemas is None:
        table_schemas = {"main": schema}
        if user_schema is not None:
            table_schemas["secondary"] = user_schema
    if database_paths is None:
        database_paths = {"main": database_path}
        if user_database_path is not None:
            database_paths["secondary"] = user_database_path

    use_table_catalog = len(table_schemas) > 1
    effective_prompt_method = "tables" if use_table_catalog else (prompt_method or "single")
    prompt_template = prompts.get_code_prompt(method=effective_prompt_method)

    available_functions = [
        func_name
        for func_name, obj in inspect.getmembers(tools)
        if inspect.isfunction(obj)
    ]
    function_docs = []
    for func_name in available_functions:
        function_docs.append(
            f"{func_name}{inspect.signature(getattr(tools, func_name))}:\n{inspect.getdoc(getattr(tools, func_name))}\n"
            + "=" * 20
            + "\n"
        )
    function_docs = "\n".join(function_docs)
    analysis_toolkit = {"enabled": False}
    if shared_analysis_tools:
        shared_docs, analysis_toolkit = _shared_analysis_tool_context()
        function_docs += "\n\n" + shared_docs

    # instantiate llm model
    llm = get_chat_model(model_name, temperature, seed)

    # create prompt
    if use_table_catalog:
        table_catalog = json.dumps(
            [
                {
                    "name": name,
                    "csv_path": database_paths.get(name),
                    "schema": table_schema,
                }
                for name, table_schema in table_schemas.items()
            ],
            indent=2,
            default=str,
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["goal", "table_catalog", "question", "function_docs"],
        )
    elif user_schema is None:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "goal",
                "schema",
                "question",
                "database_path",
                "function_docs",
            ],
        )
    else:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "goal",
                "schema",
                "question",
                "database_path",
                "function_docs",
                "user_schema",
                "user_database_path",
            ],
        )

    # Run the retry on error function
    if use_table_catalog:
        output, completions = retry_on_parsing_error(
            llm=llm,
            initial_prompt=prompt.format(
                goal=goal,
                table_catalog=table_catalog,
                question=question,
                function_docs=function_docs,
            ),
            parser=partial(_code_parser, output_folder=output_folder),
            n_retries=n_retries,
            exception_on_max_retries=False,
        )
    elif user_schema is None:
        output, completions = retry_on_parsing_error(
            llm=llm,
            initial_prompt=prompt.format(
                goal=goal,
                schema=schema,
                question=question,
                database_path=database_path,
                function_docs=function_docs,
            ),
            parser=partial(_code_parser, output_folder=output_folder),
            n_retries=n_retries,
            exception_on_max_retries=False,
        )
    else:
        output, completions = retry_on_parsing_error(
            llm=llm,
            initial_prompt=prompt.format(
                goal=goal,
                schema=schema,
                question=question,
                database_path=database_path,
                function_docs=function_docs,
                user_schema=user_schema,
                user_database_path=user_database_path,
            ),
            parser=partial(_code_parser, output_folder=output_folder),
            n_retries=n_retries,
            exception_on_max_retries=False,
        )

    # Create the output dict
    # Then, iterate over all generated plots and add them to the output dict
    output_dict = {
        "code": completions[-1],
        "prompt": str(prompt),
        "code_output": output,
        "message": output,
        "n_retries": len(completions) - 1,
        "goal": goal,
        "question": question,
        "vars": [],
        "analysis_toolkit": analysis_toolkit,
    }

    # write code to a file
    with open(f"{output_folder}/code.py", "w") as file:
        # use regex to capture the python code block
        code = extract_python_code_blocks(completions[-1])
        if not code:
            code = completions[-1]
        file.write(normalize_generated_code(code).strip())

    # Try to load the model's output files
    # TODO: We should detect errors in such files and trigger a retry
    try:
        stat = json.load(open(f"{output_folder}/stat.json", "r"))
    except Exception as e:
        print(f"Failed to load {output_folder}/stat.json", e)
        stat = {}
    try:
        x_axis = json.load(open(f"{output_folder}/x_axis.json", "r"))
    except Exception as e:
        print(f"Failed to load {output_folder}/x_axis.json", e)
        x_axis = {}
    try:
        y_axis = json.load(open(f"{output_folder}/y_axis.json", "r"))
    except Exception as e:
        print(f"Failed to load {output_folder}/y_axis.json", e)
        y_axis = {}

    # Add the plot to the final output dict
    plot_path = f"{output_folder}/plot.jpg"
    stat["type"] = "stat"
    x_axis["type"] = "x_axis"
    y_axis["type"] = "y_axis"
    plot_dict = {"name": plot_path, "type": "plot"}
    output_dict["vars"] += [
        {
            "stat": stat,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "plot": plot_dict,
        }
    ]

    return output_dict


# def root_depth_to_prompt(
#     history: Dict,
#     root: int,
#     depth: int,
#     goal: str,
#     csv_path: str,
#     results_dir: str,
# ) -> Dict:
#     """Get question from a given depth in the history and convert it to the prompt format"""
#     context = "This is a dataset of ServiceNow incidents that contains different types of failure categories"
#     import prompts

#     node = f"{root}{depth}"
#     node_output = history[node]
#     # extract question being asked at this node
#     question = node_output["question"]
#     # extract the data_df
#     data_df = pd.read_csv(csv_path)
#     prompts_dict = {}
#     # now, reconstruct the prompts used in different stages at this node
#     if root == 0 and depth == 0:
#         get_questions_prompt = prompts.GET_QUESTIONS_TEMPLATE
#         get_questions_system = prompts.GET_QUESTIONS_SYSTEM_MESSAGE
#         messages = [
#             SystemMessage(content=get_questions_system),
#             HumanMessage(
#                 content=get_questions_prompt.format(
#                     context=context,
#                     goal=goal,
#                     schema=get_schema(data_df),
#                     max_questions=3,
#                 )
#             ),
#         ]
#         prompts_dict["get_questions"] = messages
#     else:
#         prompts_dict["get_questions"] = None

#     # prompt for generating code
#     code_prompt = prompts.GENERATE_CODE_TEMPLATE
#     available_functions = [
#         func_name
#         for func_name, obj in inspect.getmembers(tools)
#         if inspect.isfunction(obj)
#     ]
#     function_docs = []
#     for func_name in available_functions:
#         function_docs.append(
#             f"{func_name}{inspect.signature(getattr(tools, func_name))}:\n{inspect.getdoc(getattr(tools, func_name))}\n"
#             + "=" * 20
#             + "\n"
#         )
#     function_docs = "\n".join(function_docs)
#     template = prompts.GENERATE_CODE_TEMPLATE
#     code_prompt = PromptTemplate(
#         template=template,
#         input_variables=[
#             "goal",
#             "schema",
#             "question",
#             "database_path",
#             "function_docs",
#         ],
#     )
#     schema = get_schema(data_df)
#     prompts_dict["generate_code"] = code_prompt.format(
#         goal=goal,
#         schema=schema,
#         question=question,
#         database_path=csv_path,
#         function_docs=function_docs,
#     )

#     output_folder = os.path.join(results_dir, node)
#     solution_dict = {
#         "code": open(os.path.join(output_folder, "code.py")).read(),
#         "prompt": str(code_prompt),
#         "code_output": "N/A",
#         "message": "N/A",
#         "n_retries": 3,
#         "goal": goal,
#         "question": question,
#         "vars": [],
#     }

#     # Extract the IDs of all generated plots
#     plot_ids = set(
#         [
#             os.path.splitext(f)[0].split("_")[-1]
#             for f in os.listdir(output_folder)
#             if any(w in f for w in ["plot_", "stat_", "x_axis_", "y_axis_"])
#         ]
#     )

#     for pid in plot_ids:
#         # Try to load the model's output files
#         # TODO: We should detect errors in such files and trigger a retry
#         try:
#             stat = json.load(open(f"{output_folder}/stat_{pid}.json", "r"))
#         except Exception as e:
#             print(f"Failed to load {output_folder}/stat_{pid}.json", e)
#             stat = {}
#         try:
#             x_axis = json.load(open(f"{output_folder}/x_axis_{pid}.json", "r"))
#         except Exception as e:
#             print(f"Failed to load {output_folder}/x_axis_{pid}.json", e)
#             x_axis = {}
#         try:
#             y_axis = json.load(open(f"{output_folder}/y_axis_{pid}.json", "r"))
#         except Exception as e:
#             print(f"Failed to load {output_folder}/y_axis_{pid}.json", e)
#             y_axis = {}

#         # Add the plot to the final output dict
#         plot_path = f"{output_folder}/plot_{pid}.jpg"
#         stat["type"] = "stat"
#         x_axis["type"] = "x_axis"
#         y_axis["type"] = "y_axis"
#         plot_dict = {"name": plot_path, "type": "plot"}
#         solution_dict["vars"] += [
#             {
#                 "stat": stat,
#                 "x_axis": x_axis,
#                 "y_axis": y_axis,
#                 "plot": plot_dict,
#             }
#         ]

#     # prompt for interpreting soln
#     interpret_prompt = prompts.INTERPRET_SOLUTION
#     insight_prompt = _build_insight_prompt(solution_dict)
#     prompts_dict["interpret_solution_prompt"] = interpret_prompt.format(
#         goal=solution_dict["goal"],
#         question=solution_dict["question"],
#         message=solution_dict["message"],
#         insights=insight_prompt,
#     )

#     # build data analysis prompt
#     data_analysis_prompt = prompts.DATA_ANALYTICS_TEMPLATE
#     da_messages = [
#         SystemMessage(content=prompts.GET_DATA_ANALYTICS_SYSTEM_MESSAGE),
#         HumanMessage(
#             content=data_analysis_prompt.format(
#                 context=context,
#                 goal=goal,
#                 schema=schema,
#                 question=question,
#                 answer=node_output["answer"]["answer"],
#                 justification=node_output["answer"]["justification"],
#                 max_questions=3,
#             )
#         ),
#     ]

#     prompts_dict["data_analysis"] = da_messages

#     # get select a follow up prompt
#     select_follow_up_prompt = prompts.SELECT_A_QUESTION_TEMPLATE
#     followup_questions_formatted = "\n".join(
#         [f"{i+1}. {q}\n" for i, q in enumerate(node_output["follow_ups"])]
#     )
#     prev_questions_formatted = "\n".join(
#         [
#             f"{i+1}. {q}\n"
#             for i, q in enumerate(
#                 output["question"]
#                 for node, output in history.items()
#                 if (int(node[0]) <= int(root) and int(node[1]) <= depth)
#             )
#         ]
#     )
#     messages = [
#         SystemMessage(content=prompts.SELECT_A_QUESTION_SYSTEM_MESSAGE),
#         HumanMessage(
#             content=select_follow_up_prompt.format(
#                 context=context,
#                 goal=goal,
#                 prev_questions_formatted=prev_questions_formatted,
#                 followup_questions_formatted=followup_questions_formatted,
#             )
#         ),
#     ]

#     prompts_dict["select_question"] = messages

#     # build the G-Eval prompt
#     curr_answer = node_output["answer"]["answer"]
#     scores_dict = json.load(open(os.path.join(results_dir, "scores.json")))
#     all_gts = [n["gt"] for gt_id, n in scores_dict.items()]

#     geval_prompt = prompts.G_EVAL_TEMPLATE
#     prompts_dict["geval_prompts"] = [
#         prompts.G_EVAL_SYSTEM_MESSAGE,
#         [geval_prompt.format(answer=curr_answer, gt_answer=gt) for gt in all_gts],
#     ]

#     return prompts_dict


def analysis_nb_to_gt(fname_notebook, include_df_head=False) -> None:
    """
    Reads all ipynb files in data_dir and parses each cell and converts it into a ground truth file.
    The ipynb files are structured as follows: code (outputs plot), then a cell with an insight dict
    """

    def _extract_metadata(nb):
        # iterate through the cells
        metadata = {}
        # extract metadata

        # extract name of the dataset from the first cell
        dname = re.findall(r"## (.+) \(Flag \d+\)", nb.cells[0].source)[0].strip()
        metadata["dataset_name"] = dname
        # extract dataset description
        description = (
            re.findall(
                r"(Dataset Overview|Description)(.+)(Your Objective|Task)",
                nb.cells[0].source,
                re.DOTALL,
            )[0][1]
            .replace("#", "")
            .strip()
        )
        metadata["dataset_description"] = description

        # extract goal and role
        goal_matches = re.findall(
            r"(?:Goal|Objective)\**:\s*(.+)", nb.cells[0].source
        )
        if not goal_matches:
            raise ValueError("Notebook metadata must contain a Goal: or Objective: line")
        metadata["goal"] = goal_matches[0].strip()
        metadata["role"] = re.findall(r"Role\**:(.+)", nb.cells[0].source)[0].strip()

        metadata["difficulty"] = re.findall(
            r"Difficulty|Challenge Level\**: (\d) out of \d", nb.cells[0].source
        )[0].strip()
        metadata["difficulty_description"] = (
            re.findall(
                r"Difficulty|Challenge Level\**: \d out of \d(.+)", nb.cells[0].source
            )[0]
            .replace("*", "")
            .strip()
        )
        metadata["dataset_category"] = re.findall(
            r"Category\**: (.+)", nb.cells[0].source
        )[0].strip()

        # Get Dataset Info
        tag = r"^dataset_path =(.+)"

        dataset_csv_path = None
        for cell in nb.cells:
            if cell.cell_type == "code":
                if re.search(tag, cell.source):
                    dataset_csv_path = (
                        re.findall(tag, cell.source)[0]
                        .strip()
                        .replace("'", "")
                        .replace('"', "")
                    )
                    break
        assert dataset_csv_path is not None
        metadata["dataset_csv_path"] = dataset_csv_path

        if include_df_head:
            metadata["df_head"] = pd.read_html(
                StringIO(cell.outputs[0]["data"]["text/html"])
            )

        # Get Dataset Info
        tag = r"user_dataset_path =(.+)"

        user_dataset_csv_path = None
        for cell in nb.cells:
            if cell.cell_type == "code":
                if re.search(tag, cell.source):
                    user_dataset_csv_path = (
                        re.findall(tag, cell.source)[0]
                        .strip()
                        .replace("'", "")
                        .replace('"', "")
                    )
                    break
        metadata["user_dataset_csv_path"] = user_dataset_csv_path

        # Get Summary of Findings
        tag = r"Summary of Findings \(Flag \d+\)(.+)"

        flag = None
        for cell in reversed(nb.cells):
            if cell.cell_type == "markdown":
                if re.search(tag, cell.source, re.DOTALL | re.IGNORECASE):
                    flag = (
                        re.findall(tag, cell.source, re.DOTALL | re.IGNORECASE)[0]
                        .replace("#", "")
                        .replace("*", "")
                        .strip()
                    )
                    break
        assert flag is not None
        metadata["flag"] = flag

        return metadata

    def _parse_question(nb, cell_idx):
        qdict = {}
        qdict["question"] = (
            re.findall(
                r"Question( |-)(\d+).*:(.+)", nb.cells[cell_idx].source, re.IGNORECASE
            )[0][2]
            .replace("*", "")
            .strip()
        )

        if nb.cells[cell_idx + 2].cell_type == "code":
            # action to take to answer the question
            assert nb.cells[cell_idx + 1].cell_type == "markdown"
            qdict["q_action"] = nb.cells[cell_idx + 1].source.replace("#", "").strip()
            assert nb.cells[cell_idx + 2].cell_type == "code"
            qdict["code"] = nb.cells[cell_idx + 2].source
            # extract output plot. Note that this image data is in str,
            # will need to use base64 to load this data

            qdict["plot"] = nb.cells[cell_idx + 2].outputs
            # loop as there might be multiple outputs and some might be stderr
            for o in qdict["plot"]:
                if "data" in o and "image/png" in o["data"]:
                    qdict["plot"] = o["data"]["image/png"]
                    break

            # extract the insight
            try:
                qdict["insight_dict"] = json.loads(nb.cells[cell_idx + 4].source)
            except Exception as e:
                # find the next cell with the insight dict
                for cell in nb.cells[cell_idx + 3 :]:
                    try:
                        qdict["insight_dict"] = json.loads(cell.source)
                        break
                    except Exception as e:
                        continue

        else:
            # print(f"Found prescriptive insight in {fname_notebook}")
            qdict["insight_dict"] = {
                "data_type": "prescriptive",
                "insight": nb.cells[cell_idx + 1].source.strip(),
                "question": qdict["question"],
            }
        return qdict

    def _parse_notebook(nb):
        gt_dict = _extract_metadata(nb)

        # extract questions, code, and outputs
        que_indices = [
            idx
            for idx, cell in enumerate(nb.cells)
            if cell.cell_type == "markdown"
            and re.search(r"Question( |-)\d+", cell.source, re.IGNORECASE)
        ]
        gt_dict["insights"] = []
        for que_idx in que_indices:
            gt_dict["insights"].append(_parse_question(nb, que_idx))
        return gt_dict

    # Convert the notebook to a ground truth file
    if not fname_notebook.endswith(".ipynb"):
        raise ValueError("The file must be an ipynb file")
    else:
        # extract dataset id from flag-analysis-i.ipynb using re
        fname_json = fname_notebook.replace(".ipynb", ".json")

        with open(fname_notebook, "r") as f:
            notebook = nbformat.read(f, as_version=4)
        gt_dict = _parse_notebook(notebook)

    return gt_dict


def get_chat_model(model_name, temperature=0, seed=None):
    if "gpt" in model_name:
        client = OpenAI(api_key=OPENAI_API_KEY)
        api_model_name = model_name
        provider = "openai"
    elif model_name.lower() == "qwen3.8-flash":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for Qwen3.8 Flash")
        client = OpenAI(
            api_key=api_key,
            base_url=(os.getenv("DASHSCOPE_BASE_URL") or os.getenv("QWEN_BASE_URL")
                      or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
            timeout=180,
        )
        api_model_name = "qwen3.8-flash"
        provider = "qwen"
    elif model_name.lower() == "deepseek-v4.1-flash":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for DashScope DeepSeek")
        client = OpenAI(
            api_key=api_key,
            base_url=(os.getenv("DASHSCOPE_BASE_URL") or os.getenv("QWEN_BASE_URL")
                      or "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            timeout=180,
        )
        api_model_name = "deepseek-v4.1-flash"
        provider = "dashscope_deepseek"
    elif model_name.lower() == "mimo-v2.6-flash":
        api_key = os.getenv("XIAOMI_API_KEY")
        if not api_key:
            raise ValueError("XIAOMI_API_KEY is required for MiMo-V2.6-Flash")
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.xiaomimimo.com/v1",
            timeout=180,
        )
        api_model_name = "mimo-v2.6-flash"
        provider = "xiaomi"
    elif model_name.lower() in {
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini_3_5_flash_lite",
    }:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini models")
        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
        )
        api_model_name = "gemini-3.5-flash" if model_name.lower() == "gemini-3.5-flash" else "gemini-3.5-flash-lite"
        provider = "gemini"
    elif model_name.lower() in {
        "deepseek-flash",
        "deepseek_flash",
        "deepseek-v4.1-flash",
        "deepseek_v4_1_flash",
    }:
        # Official DeepSeek OpenAI-compatible endpoint. This branch lets the
        # InsightBench workflow use the exact same endpoint/model as the
        # native TrajStory framework during paired comparisons.
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for official DeepSeek models")
        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        api_model_name = "deepseek-flash"
        provider = "deepseek"
    elif model_name.lower() in {
        "glm-5.3-flash",
        "glm_5_3_flash",
    }:
        api_key = os.getenv("GLM_API_KEY") or os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("GLM_API_KEY or ZHIPU_API_KEY is required for official GLM models")
        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv(
                "GLM_BASE_URL", os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
            ),
        )
        api_model_name = "glm-5.3-flash"
        provider = "zhipu"
    elif model_name.lower() in {
        "deepseek-v4-pro",
        "deepseek_v4_pro",
        "deepseek-v4-flash-0731",
        "deepseek_v4_flash_0731",
    }:
        # Together provides DeepSeek V4 models through an OpenAI-compatible
        # endpoint. Keep short benchmark labels while using the provider's
        # canonical model identifiers on the wire.
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY is required for DeepSeek V4 on Together AI")
        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),
        )
        if "flash" in model_name.lower():
            api_model_name = "deepseek-ai/DeepSeek-V4-Flash-0731"
        else:
            api_model_name = "deepseek-ai/DeepSeek-V4-Pro"
        provider = "together"
    elif "llama" in model_name.lower() or "meta-llama" in model_name.lower():
        # InsightBench hosts LLaMA through an OpenAI-compatible vLLM server.
        # Keep the short benchmark label while mapping it to the served model id.
        base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.ai/v1")
        api_key = os.getenv("TOGETHER_API_KEY", "EMPTY")
        api_model_name = os.getenv(
            "TRAJSTORY_LLAMA_MODEL_NAME", "meta-llama/Meta-Llama-3-70B"
        )
        client = OpenAI(api_key=api_key, base_url=base_url)
        provider = "llama"
    elif model_name.lower() in {
        "google/gemma-4-31b-it",
        "gemma-4-31b-it",
    }:
        # Together AI serves Gemma 4 31B IT through its OpenAI-compatible API.
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY is required for Gemma 4 on Together AI")
        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("TOGETHER_BASE_URL", "https://api.together.ai/v1"),
        )
        api_model_name = "google/gemma-4-31B-it"
        provider = "together"
    else:
        raise ValueError(f"Unsupported generation model: {model_name}")

    if provider in {"openai", "together", "llama", "gemini", "deepseek", "zhipu", "qwen", "dashscope_deepseek", "xiaomi"}:
        request_kwargs = {"model": api_model_name}
        is_openai_reasoning = provider == "openai" and api_model_name.startswith(("gpt-5.6", "gpt-6-luna"))
        reasoning_effort = os.getenv("TRAJSTORY_OPENAI_REASONING_EFFORT")
        if provider == "qwen":
            reasoning_effort = os.getenv("TRAJSTORY_QWEN_REASONING_EFFORT")
        # Gemini 3.5 Flash-Lite deprecates sampling parameters. Keep the
        # benchmark seed in run metadata, but do not send it to Gemini.
        # GPT-5.6 / GPT-6 Luna sampling parameters are sent only in non-reasoning mode;
        # reasoning runs keep the service default and omit temperature.
        if (provider != "gemini" or api_model_name == "gemini-3.5-flash") and (
            not is_openai_reasoning or reasoning_effort == "none"
        ):
            request_kwargs["temperature"] = temperature
        if seed is not None and provider not in {"gemini", "deepseek", "zhipu", "dashscope_deepseek"}:
            request_kwargs["seed"] = int(seed)
        if provider in {"deepseek", "zhipu", "dashscope_deepseek"}:
            request_kwargs["max_tokens"] = 16_000
        elif provider == "xiaomi":
            request_kwargs["max_completion_tokens"] = 16_000
        elif provider == "gemini" and api_model_name == "gemini-3.5-flash":
            request_kwargs["max_tokens"] = 16_000
            if reasoning_effort is not None:
                request_kwargs["reasoning_effort"] = reasoning_effort
        elif is_openai_reasoning or provider == "qwen":
            request_kwargs["max_completion_tokens"] = 16_000
            if reasoning_effort:
                request_kwargs["reasoning_effort"] = reasoning_effort
        if provider == "qwen" and reasoning_effort:
            request_kwargs["extra_body"] = {"enable_thinking": reasoning_effort != "none"}
        if provider == "qwen":
            qwen_seed = 0 if seed is None else seed
            if type(qwen_seed) is not int or not 0 <= qwen_seed < 2**31:
                raise ValueError("seed must be an integer in [0, 2**31 - 1]")
            request_kwargs["seed"] = qwen_seed
            if temperature == 0 and reasoning_effort == "none":
                request_kwargs.setdefault("extra_body", {})["top_k"] = 1
        if provider == "dashscope_deepseek" and seed is not None:
            if type(seed) is not int or not 0 <= seed < 2**31:
                raise ValueError("seed must be an integer in [0, 2**31 - 1]")
            request_kwargs["seed"] = seed
        if provider == "deepseek" and seed is not None:
            if type(seed) is not int or not 0 <= seed < 2**31:
                raise ValueError("seed must be an integer in [0, 2**31 - 1]")
            request_kwargs["seed"] = seed
        if provider == "deepseek" and reasoning_effort == "none":
            request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if provider == "zhipu":
            if reasoning_effort not in {"low", "high", "max"}:
                raise ValueError("GLM-5.3-Flash requires reasoning_effort low, high, or max")
            request_kwargs["reasoning_effort"] = reasoning_effort
            request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        if provider == "xiaomi":
            if reasoning_effort != "none":
                raise ValueError("MiMo benchmark requires reasoning_effort='none'")
            request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if provider == "dashscope_deepseek":
            thinking_mode = os.getenv("TRAJSTORY_DASHSCOPE_DEEPSEEK_THINKING")
            if thinking_mode not in {"enabled", "disabled"}:
                raise ValueError("TRAJSTORY_DASHSCOPE_DEEPSEEK_THINKING must be enabled or disabled")
            request_kwargs["extra_body"] = {"enable_thinking": thinking_mode == "enabled"}
        llm = (
            lambda content: client.chat.completions.create(
                **request_kwargs,
                messages=[{"role": "user", "content": content}],
            )
            .choices[0]
            .message.content
        )
        if provider in {"qwen", "deepseek", "dashscope_deepseek", "gemini", "xiaomi", "zhipu"} or api_model_name.startswith("gpt-6-luna"):
            def llm(content):
                response = client.chat.completions.create(
                    **request_kwargs, messages=[{"role": "user", "content": content}],
                )
                if os.getenv("TRAJSTORY_LLM_AUDIT_PATH"):
                    from baseline_support.llm_audit import record_completion
                    llm.last_response_metadata = record_completion(
                        {**request_kwargs, "messages": [{"role": "user", "content": content}]},
                        response.model_dump(),
                    )
                return response.choices[0].message.content

    return llm


class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


if __name__ == "__main__":
    # dataset_id = 1
    # results_dir = f"./.tmp/outputs_no_goal/gpt-4-turbo-2024-04-09/{dataset_id}/"
    # data_dir = "/mnt/cba/data/servicenow_incidents/flags"
    # goal = json.load(open(os.path.join(data_dir, f"gt_flag_{dataset_id}.json")))["Goal"]
    # history = json.load(open(os.path.join(results_dir, "history.json")))
    # root_depth_to_prompt(
    #     history=history,
    #     root=1,
    #     depth=3,
    #     goal=goal,
    #     csv_path=os.path.join(data_dir, f"flag-{dataset_id}.csv"),
    #     results_dir=results_dir,
    # )

    analysis_nb_to_gt("/mnt/home/projects/research-cba/.tmp/new_notebooks")
