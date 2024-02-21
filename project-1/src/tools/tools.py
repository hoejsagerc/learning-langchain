from langchain.serpapi import SerpAPIWrapper

import os


def get_profile_url(text: str) -> str:
    """Searches for LinkedIn Profile Pages."""
    search = SerpAPIWrapper(serpapi_api_key=os.environ.get("SERPAPI_API_KEY"))
    res = search.run(f"{text}")
    return res
