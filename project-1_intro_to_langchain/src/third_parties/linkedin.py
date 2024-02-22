import os
import requests

# Langchain can use the docstrings in the functions to choose wether to use the function or not.
# Therefore we can use the docstrings to tell Langchain what exactly this function can be used for.

# For the purpose of training i can use this url to simulate the scraping of LinkedIn api
# url = "https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json"


def scrape_linkedin_profile(linkedin_profile_url: str) -> str:
    """Scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""

    # Real code => not used because of missing credits on proxyCurl
    # api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    # header_dic = {"Authorization": f"Bearer {os.environ.get('PROXYCURL_API_KEY')}"}

    # response = requests.get(
    #     api_endpoint, params={"url": linkedin_profile_url}, headers=header_dic
    # )

    api_endpoint = "https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json"

    response = requests.get(api_endpoint)

    # Manipulating the response to remove all empty fields in the json.
    # The reason for this is to save tokens for the llm call
    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }

    # Removing the profile pic url from groups
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data
