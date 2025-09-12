#!/usr/bin/env python3.12
# coding: utf-8

# # Chains in LangChain
# 
# #### Chain combines a large language model with a prompt. That in-turn can be used to carryout a sequence of tasks on text or on your data by putting them together in a chain format. 
# 
# ## Outline
# 
# * LLMChain
# * Sequential Chains
#   * SimpleSequentialChain
#   * SequentialChain
# * Router Chain

# In[1]:


import warnings
warnings.filterwarnings('ignore')

# In[2]:


# load environment variables 
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# Note: LLM's do not always produce the same results. When executing the code in your notebook, you may get slightly different answers that those in the lecture.

# In[3]:


# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

# In[4]:


#!pip install pandas

# In[5]:


# load the dataset 
# power of chain is that you can use them on many inputs simultaniously 
import pandas as pd
df = pd.read_csv('../data/Data.csv')

# In[6]:


df.head(10)
# observe the product column and the review column 
# each row is a product and its review 

# In[7]:


df.shape

# ## LLMChain
# 
# Quite simple: its just a combination of the llm and the prompt 

# In[8]:


# lang chain abstraction for OpenAI model -> the llm
from langchain.chat_models import ChatOpenAI

# `ChatPromptTemplate` -> this is the prompt 
from langchain.prompts import ChatPromptTemplate

# LLMChain 
from langchain.chains import LLMChain

# In[9]:


# initialise the language model that we want to use -> High temp - for fun desc.
llm = ChatOpenAI(temperature=0.9, model=llm_model)

# In[10]:


# takes the product as an input -> asks LLM -> returns 
# the best suitable company name which makes that product. 
# company name prediction 
prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

# In[11]:


# combine prompt and llm into a chain 
# this is what we call an LLM chain 
chain = LLMChain(llm=llm, prompt=prompt)

# In[ ]:


# the above chain will let us run through the prompt, now. 
product = "Queen Size Sheet Set"
chain.run(product)

# under the hood it will format the prompt and 
# then it will pass the whole prompt into the llm

# LLM chain is the most basic type of chain and that is going to be used a lot in the applications. 
# ### Pause here and try to run this with different product names to run the chain

# ## SimpleSequentialChain
# 
# Slide #15
# 
# Runs one after another as a sequence. 

# In[13]:


# importing the `SimpleSequentialChain` library
from langchain.chains import SimpleSequentialChain

# important note: This works well when we have subchains 
# that expect only one input and return only one output.

# In[14]:


llm = ChatOpenAI(temperature=0.9, model=llm_model)

# product is already defined above as follows:
# product = "Queen Size Sheet Set"

# prompt template 1: 
# input: product 
# output: company name 
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

# Chain 1
chain_one = LLMChain(llm=llm, prompt=first_prompt)

# In[15]:


# input: company name 
# output: description of that company
# prompt template 2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    company:{company_name}"
)
# chain 2
chain_two = LLMChain(llm=llm, prompt=second_prompt)

# In[16]:


# create an object of SimpleSequentialChain with chain_one and chain_2 as a sequence 
overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True
                                            )

# In[17]:


# run this chain over any product / product desc. 
overall_simple_chain.run(product)

# SimpleSqeuentialChain works well when there is a single input and a single output. 
# 
# What if there are multiple inputs or outputs?
# 
# Solution: 👇🏻 SequentialChain

# ## SequentialChain
# 
# Slide #16

# In[18]:


from langchain.chains import SequentialChain

# In[19]:


# chain 1 -- take a review and translate it into english 
llm = ChatOpenAI(temperature=0.9, model=llm_model)

# prompt template 1: translate to english

first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)
# chain 1: input= Review and output= English_Review
# input: review 
# output: english translation of the review 
chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                     output_key="English_Review"
                    )

# In[20]:


# chain 2 -- create a summary of that review in 1 sentence. 
# this will use the previously generated English Review 

second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)
# chain 2: input= English_Review and output= summary of that review in one sentence. 
chain_two = LLMChain(llm=llm, prompt=second_prompt, 
                     output_key="summary"
                    )

# In[21]:


# chain 3 -- detect the language of the original review 

# prompt template 3: translate to english
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)
# chain 3: input= Review and output= language
chain_three = LLMChain(llm=llm, prompt=third_prompt,
                       output_key="language"
                      )


# In[22]:


# chain 4 -- it will take in multiple inputs viz. `summary` from chain-2 and `language` from chain-3 and 
# its going to ask the follow up response for the summary in the specified language. 

# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)
# chain 4: input= summary, language and output= followup_message
chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                      output_key="followup_message"
                     )

# It is important to note that the input-keys and output-keys needs to be very precise. 
# e.g. 
# - chain - 1 takes `Review` -- passed in the start and `output_key = English_review` 
# - chain - 2 takes `English_review` -- and `output_key = summary`
# - chain - 3 takes `Review` -- and `output_key=language`
# - chain - 4 takes `summary` + `language` -- and `output_key=followup_message`
# 
# Note: becareful with the variable names here

# In[23]:


# overall_chain: input= Review 
# and output= English_Review,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","followup_message","language"],
    verbose=True
)

# In[24]:


review = df.Review[5]
overall_chain(review)

# You should take a pause here and try putting in different inputs by yourself
# 
# Task: how can we add english_followup_message into the output to understand what was sent by the LLM to the reviewer?

# ## Router Chain
# 
# Use this when you want to leverage choices based on the branching. 
# 
# Slide #18

# In[25]:


# defining subject matter experts as prompt templates 

# good for answering `physics` questions 
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""

# good for answering `Mathematics` questions 
math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

# good for answering `History` questions 
history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""


# good for answering `Computer Science` questions 
computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 

Here is a question:
{input}"""

# In[26]:


# this information is going to be passed to the router chain 
# where router_chain will be deciding when to use which subject matter experts

prompt_infos = [
    {
        "name": "physics", 
        "description": "Good for answering questions about physics", 
        "prompt_template": physics_template
    },
    {
        "name": "math", 
        "description": "Good for answering math questions", 
        "prompt_template": math_template
    },
    {
        "name": "History", 
        "description": "Good for answering history questions", 
        "prompt_template": history_template
    },
    {
        "name": "computer science", 
        "description": "Good for answering computer science questions", 
        "prompt_template": computerscience_template
    }
]

# In[27]:


# MultiPromptChain is required when we want to route between multiple different prompt templates 
from langchain.chains.router import MultiPromptChain

# LLMRouterChain uses LLM to route between the different subchains -- hence "name" and "description" will be used 
# RouterOutputParser parsing the LLM output in a python dict which can be used downstream to determine which chain to use and what to input to that chain should be. 
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

from langchain.prompts import PromptTemplate

# In[28]:


# import and define the language model that we will use
llm = ChatOpenAI(temperature=0, model=llm_model)

# In[32]:


# we are creating `destination_chains` and `destinations_str` which will be used by router later. 

# IMP: Understand, `destination_chains` as available targets that the router can pick from

destination_chains = {}
for p_info in prompt_infos:
    # take a value of a `prompt_template` key from the `prompt_infos` dict 
    prompt_template = p_info["prompt_template"]
    # create a promp using `prompt_template`
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    # create a `chain` using `llm` and `prompt`
    chain = LLMChain(llm=llm, prompt=prompt)
    # # take the `name`` from the `prompt_infos` dict 
    # name = p_info["name"]
    # create a key with the `name` and keep the chain in the same key
    destination_chains[p_info["name"]] = chain  

# the menu of choices (names + descriptions) shown to the router to guide selection.
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# In[33]:


destinations_str

# In[34]:


# this is the default chain which will be used in case when no experts are matched 
# in that case we directly assign the prompt `{input}` to the llm 
# decision to route the {input} to the default chain based llm is also taken by router
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

# In[35]:


# template which is used to route inputs between different chains 

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ "DEFAULT" or name of the prompt to use in {destinations}
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: The value of “destination” MUST match one of \
the candidate prompts listed below.\
If “destination” does not fit any of the specified prompts, set it to “DEFAULT.”
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

# In[36]:


# Goal: To create a `router_chain` using `llm`` and `router_prompt`
# 1. create a `router_template` using `destinations_str`
# 2. create a `router_prompt` using `destination_chains`, input, and RouterOutputParser class. 

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# In[37]:


# Last step before we infer: creating a overall `chain` object to put everything together. 
chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, 
                         verbose=True
                        )

# In[38]:


chain.run("What is black body radiation?")

# In[39]:


chain.run("what is 2 + 2")

# In[40]:


chain.run("Why does every cell in our body contain DNA?")

# Task: add few more subject templates and modify this Router Chain to make it work with more different subjects
