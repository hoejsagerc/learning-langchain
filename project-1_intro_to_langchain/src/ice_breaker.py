from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from parsers.person_intel_parser import PersonIntel, person_intel_parser

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from typing import Tuple


import os


def ice_break(name: str) -> Tuple[PersonIntel, str]:
    # Using the linkedin_lookup_agent to get the linkedin profile url
    # This agent uses SerpApi to make a google search for the linkedin profile url
    linkedin_profile_url = linkedin_lookup_agent(name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url)

    # Creating the prompt template Text
    summary_template = """
    given the inmformation {information} about a person I want you to create:
    1. A short summary
    2. Two interesting fact about the person
    3. A topic that may interest them
    4. 2 creative ice breakers to open a conversation with the them
    \n{format_instructions}
    """

    # Creating the prompt template from the template text
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            # here we are using a parser which basicly is just a pydantic model.
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    # Defining the llm to be used in the chain
    llm = ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"]
    )

    # Creating the chain
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # invoking the chain
    # result = chain.invoke(input={"information": linkedin_data})
    result = chain.run(information=linkedin_data)

    return person_intel_parser.parse(result), linkedin_data.get("profile_pic_url")


if __name__ == "__main__":
    load_dotenv()
    result = ice_break("Eden Marco")
    print(result)
