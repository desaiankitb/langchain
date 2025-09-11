"""LangChain: Models, Prompts and Output Parsers

This script mirrors the Jupyter notebook ``A1-Model_prompt_parser.ipynb``
using standard Python code. It requires Python 3.12 and an ``OPENAI_API_KEY``
set in the environment or in a ``.env`` file.
"""

from __future__ import annotations

import datetime
import os

import openai
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate

# Load API key
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

# Handle model deprecation
current_date = datetime.datetime.now().date()
TARGET_DATE = datetime.date(2025, 8, 18)
llm_model = "gpt-3.5-turbo" if current_date > TARGET_DATE else "gpt-3.5-turbo-0301"


def get_completion(prompt: str, model: str = llm_model) -> str:
    """Return the completion for a given prompt using ChatCompletion."""
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0.0
    )
    return response.choices[0].message["content"]


def pirate_to_calm_english() -> None:
    customer_email = """\
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!\
"""

    style = """American English \
in a calm and respectful tone"""

    prompt = f"""Translate the text \
that is delimited by triple backticks
into a style that is {style}.
text: ```{customer_email}```
"""
    print(prompt)
    response = get_completion(prompt)
    print(response)


def prompt_templates_and_translation() -> None:
    chat = ChatOpenAI(temperature=0.0, model=llm_model)
    template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
    prompt_template = ChatPromptTemplate.from_template(template_string)
    _ = prompt_template.messages[0].prompt.input_variables

    customer_style = """American English \
in a calm and respectful tone"""

    customer_email = """\
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!\
"""

    customer_messages = prompt_template.format_messages(
        style=customer_style, text=customer_email
    )
    print(type(customer_messages))
    print(type(customer_messages[0]))
    print(customer_messages[0])
    customer_response = chat(customer_messages)
    print(customer_response.content)

    service_reply = """\
Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!\
"""

    service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""

    service_messages = prompt_template.format_messages(
        style=service_style_pirate, text=service_reply
    )
    print(service_messages[0].content)
    service_response = chat(service_messages)
    print(service_response.content)


def structured_output_example() -> None:
    chat = ChatOpenAI(temperature=0.0, model=llm_model)
    customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.\
"""

    review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""
    prompt_template = ChatPromptTemplate.from_template(review_template)
    messages = prompt_template.format_messages(text=customer_review)
    response = chat(messages)
    print(response.content)
    print(type(response.content))
    try:
        print(response.content.get("gift"))
    except AttributeError as exc:
        print(f"Error: {exc}")

    gift_schema = ResponseSchema(
        name="gift",
        description=(
            "Was the item purchased as a gift for someone else? "
            "Answer True if yes, False if not or unknown."
        ),
    )
    delivery_days_schema = ResponseSchema(
        name="delivery_days",
        description=(
            "How many days did it take for the product to arrive? "
            "If this information is not found, output -1."
        ),
    )
    price_value_schema = ResponseSchema(
        name="price_value",
        description=(
            "Extract any sentences about the value or price, "
            "and output them as a comma separated Python list."
        ),
    )
    response_schemas = [gift_schema, delivery_days_schema, price_value_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    print(format_instructions)

    review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product
 to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""
    prompt = ChatPromptTemplate.from_template(template=review_template_2)
    messages = prompt.format_messages(
        text=customer_review, format_instructions=format_instructions
    )
    print(messages[0].content)
    response = chat(messages)
    print(response.content)
    output_dict = output_parser.parse(response.content)
    print(output_dict)
    print(type(output_dict))
    print(output_dict.get("price_value"))


def main() -> None:
    print(get_completion("What is 1+1?"))
    pirate_to_calm_english()
    prompt_templates_and_translation()
    structured_output_example()


if __name__ == "__main__":
    main()
