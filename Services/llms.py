import openai
from groq import Groq
class Llms:

    def __init__(self,api_key, llm_model = 'gpt-4o-mini', prompt = "you are a helpful assistant"):
        self.llm_model = llm_model
        self.prompt = prompt
        if "gpt" in llm_model:
            openai.api_key = api_key
        if "llama" in llm_model or "mixtral" in llm_model:
            self.groq_client = Groq(api_key=api_key)
            
            
    def generate_response(self, query):
        if "gpt" in self.llm_model:
            completion = openai.chat.completions.create(
                messages = [
                    {"role":"assistant", "content":self.prompt},
                    {"role":"user", "content":query}
                    ]
            )
        elif "llama" in self.llm_model:
            completion = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages = [
                    {"role":"assistant", "content":self.prompt},
                    {"role":"user", "content":query}
                    ]
            )
        elif "mixtral" in self.llm_model:
            completion = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages = [
                    {"role":"assistant", "content":self.prompt},
                    {"role":"user", "content":query}
                    ]
            )
        else:
            return "Model not supported"
        return completion.choices[0].message.content
        