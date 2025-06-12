import openai
import json
import os


def generate_prompt_with_problem_statement(api_key, problem_statement, type):
    system_prompt = """ Given any problem statement and a type, generate a detailed and context-rich prompt that can be used by an 
    AI assistant to provide a comprehensive and helpful response. The generated prompt should:

            Instruct the assistant on how to approach and answer the problem effectively.
            use "type" to create a prompt according to use case, type will either be 'RAG' or 'Simple LLM Bot'
            if type is 'RAG' then user will provide his own data and question and the question will be answered using that data
            DONT say I dont know, instead chat with the user if the data provided by user does not relate to the question.
            If type is 'simple LLM Bot' then the conversation will be with AI assistant on any topic mentioned in problem statement.

    return your response in JSON format:
    {{
        prompt : <your prompt>
    }}
    
    """
    openai.api_key = api_key
    response = openai.chat.completions.create(
                response_format={"type":"json_object"},
                model="gpt-4o-mini-2024-07-18",
                messages = [
                    {"role":"assistant", "content":system_prompt},
                    {"role":"user", "content":f""" problem_statement:{problem_statement}, type:{type}"""}
                    ]
            )

    json_response = json.loads(response.choices[0].message.content)
    prompt = json_response['prompt']
    
    return prompt

