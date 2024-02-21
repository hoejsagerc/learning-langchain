from dotenv import load_dotenv
from typing import Union, List
from langchain.tools.render import render_text_description
from langchain.agents import tool, Tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import os

from callbacks import AgentCallbackHandler


load_dotenv()


# langchain agent automatic tool import. can be used on a function.
# will automatically read the parmeters, returns and description
@tool
def get_text_length(text: str) -> int:
    """Return the length of a text by characters"""
    print(f"get_text_length => {text}")
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool

    raise ValueError(f"Tool with name {tool_name} not found in {tools}")


if __name__ == "__main__":
    tools = [get_text_length]

    # This prompt has been found on the langchain hub site:
    # https://smith.langchain.com/hub/hwchase17/react
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # we need to make a stop token after the observation to avoid the model to to halocinate
    llm = ChatOpenAI(
        temperature=0,
        stop=["Observation"],
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-3.5-turbo",
        callbacks=[AgentCallbackHandler()],
    )

    # The intermediate steps will be stored here, for tracking the steps
    intermediate_steps = []

    # LCEL => Langchain Expression Language allows the pipe to chain things together
    # The pipe operator will take the output of the left side and pass it as the input to the right side
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )


    # Setting the agent steps in a while loop to keep reasoning until the agent finishes
    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        # the lambda will take the value of the dict added when running invoke()
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the lenght in characters of the text DOG?",
                "agent_scratchpad": intermediate_steps,
            }
        )
        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = (
                agent_step.tool
            )  # The tool that the agent suggestion should be used
            tool_to_use = find_tool_by_name(
                tools, tool_name
            )  # Checking that the tool exists in our tools list
            tool_input = (
                agent_step.tool_input
            )  # Taking the input that the agent suggested we use together with the tool

            observation = tool_to_use.func(
                str(tool_input)
            )  # Running the tool with the input and getting an observation
            print(observation)

            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
