# learning-langchain

Repository for learning langchaing, using udemy course: LangChain - Develop LLM powered applications with LangChain

</br>

## What is LangChain

Langchain is a framework for combining tasks together or "Chaining" tasks together
for creating autonomous tasks driven by an LLM.

You can cobine you chain to comunicate with different api's such as google, linkedin,
wikipidia or anything else and then take the answer trough an LLM to create a clever answer.

</br>

## Setting up the project

create a new virtual env.

```powershell
python -m venv .venv
```

activate the environment

```powershell
. .venv/scripts/activate
```

now we need to install the langchain library:

create a new file requirements.txt

add langchain => into the file and install it by running


```powershell
python -m pip install -r requirements.txt
```

### Setting up environment variables

create a new file inside the project-1/ named .env

you will need the package python-dotenv => this is also noted in the requirements.txt

you can now run the following in your code: 

```python
from dotenv import load_dotenv
import os


if __name__ == "__main__":
    load_dotenv()
    print("Hello, LangChain!")
    print(os.environ["COOL_API_KEY"])
```

</br>

## LangChain

Documentation: https://python.langchain.com/docs/get_started/introduction

</br>

## Prompt && PromptTemplates

a prompt is an input to the LLM. The LLM can then act on the prompt.

When working with the LLM programaticly you can change the prompt by accepting parameters
instead of hardcoding the prompt.

A PromptTemplate is a wrapper class around the prompt, giving the possibility for creating parameters
to your prompt.


</br>

## Chat Models

This is a wrapper around the LLM and gives the way for the developer to 
work with the LLM. You can think about it as text-in => text-out.


</br>


## LLM Chain

Chains allow us to combine mutliple components together to be able to create one single
application. Yoy can increase the complexity by combining multiple chains together.


</br>

## Simple chain

```python
if __name__ == "__main__":
    load_dotenv()

    print("Hello, LangChain!")

    # Creating the input parameter for the prompt
    information = """
    Elon Reeve Musk (/ˈiːlɒn/; EE-lon; born June 28, 1971) is a businessman and investor. He is the founder, chairman, CEO, and CTO of SpaceX; angel investor, CEO, product architect, and former chairman of Tesla, Inc.; owner, chairman, and CTO of X Corp.; founder of the Boring Company and xAI; co-founder of Neuralink and OpenAI; and president of the Musk Foundation. He is the second wealthiest person in the world, with an estimated net worth of US$232 billion as of December 2023, according to the Bloomberg Billionaires Index, and $182.6 billion according to Forbes, primarily from his ownership stakes in Tesla and SpaceX.[5][6][7]

    A member of the wealthy South African Musk family, Elon was born in Pretoria and briefly attended the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania, and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University, but dropped out after two days and, with his brother Kimbal, co-founded online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999, and, that same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal.

    In October 2002, eBay acquired PayPal for $1.5 billion, and that same year, with $100 million of the money he made, Musk founded SpaceX, a spaceflight services company. In 2004, he became an early investor in electric vehicle manufacturer Tesla Motors, Inc. (now Tesla, Inc.). He became its chairman and product architect, assuming the position of CEO in 2008. In 2006, Musk helped create SolarCity, a solar-energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year, Musk co-founded Neuralink—a neurotechnology company developing brain–computer interfaces—and the Boring Company, a tunnel construction company. In 2022, he acquired Twitter for $44 billion. He subsequently merged the company into newly created X Corp. and rebranded the service as X the following year. In March 2023, he founded xAI, an artificial intelligence company.
        """

    # Creating the prompt template Text
    summary_template = """
    given the inmformation {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    # Creating the prompt template from the template text
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # Defining the llm to be used in the chain
    llm = ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"]
    )

    # Creating the chain
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # invoking the chain and printing the result from our prompt
    res = chain.invoke(input={"information": information})
    print(res)
```

</br>

## Agents

if we for example want to recieve current data, the llm is trained up to a certain date.
so if we want current data then we need to enroll an agents which can collect new data or 
do more or less anything you want to do.

an agent is using tools.

a tool is a connection to external services. but an agent is smarter than just doing an
api call. It used the LLM to perform actions such as an api call. This could be that you 
have a function with a docstring explaining what something might do. The read the docstring
and used this function for collecting the data it thinks are relevant to solve that task.


## Retrieval Augmentation

If you have a large amount of data such as text files. You can split up the data into multiple chunks
and then connect the chunks together in a logical way.

## Embeddings

An embedding is a vector space meaning that a model can be converted into a text space.

You can then have an Embedding Model. This model is designed to take multiple objects and then
convert them into objects as vectors.

The model usually works as a black box, meaning we dont have any insights into how they work.

when sentences are converted into embeddings they will be very close to each other if they look
much like each other.

Example:

```plaintext
Sentence 1:
- I want to order an extra large coffee

Sentence 2:
- I'll have a tall coffee

Sentence 3:
- Quiero pedir cafe extra grande
```

These three sentences will be located very close to each other in the vector space, since the
meaning of the sentences is very much a like.

Now the language does not change how close they will be.

You can calculate how close two embeddings will be to eachother.
