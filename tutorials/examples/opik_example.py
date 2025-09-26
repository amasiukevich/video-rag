import openai
from opik import configure, track 

import dotenv
dotenv.load_dotenv('../.env')

configure()

client = openai.OpenAI()

llm_model = "gpt-4.1-mini"

@track 
def retrieve_context(input_text):
    return [
        "What specific information are you looking for?",
        "How can I assist you with your interests today?",
        "Are there any topics you'd like to explore or learn more about?",
    ]


@track 
def generate_response(input_text, context):
    full_prompt = f"If the user asks a question that is not specific, use the context to provide a relevant response.\nContext: {', '.join(context)}\nUser: {input_text}\nAI:"
    response = client.chat.completions.create(
        model=llm_model, messages=[{"role": "user", "content": full_prompt}]
    )
    return response.choices[0].message.content


context = retrieve_context("Hi how are you?")
print(generate_response("Hi how are you?", context))