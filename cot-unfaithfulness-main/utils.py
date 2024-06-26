

from time import sleep
import datetime
import glob
import json
import datetime
import os
import traceback

import anthropic
# import cohere
from pyrate_limiter import Duration, RequestRate, Limiter
from openai import OpenAI

SEP = "\n\n###\n\n"

client = OpenAI(api_key="")

OAI_rate = RequestRate(50, Duration.MINUTE)
limiter = Limiter(OAI_rate)


def add_retries(f):

    def wrap(*args, **kwargs):
        max_retries = 3
        num_retries = 0
        while True:
            try:
                result = f(*args, **kwargs)
                return result
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except KeyError:
                raise KeyError
            except Exception as e:
                print("Error: ", traceback.format_exc(), "\nRetrying in ", num_retries * 2, "seconds")
                if num_retries == max_retries:
                    traceback.print_exc()
                    return {"completion": traceback.format_exc()}
                num_retries += 1
                sleep(num_retries * 2)
            
    return wrap

@add_retries
@limiter.ratelimit('identity', delay=True)
def generate(prompt, n=1, model="text-davinci-003", max_tokens=256, logprobs=None, temperature=.7):
    return client.completions.create(
        model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, n=n, logprobs=logprobs).choices[0].text

@add_retries
@limiter.ratelimit('identity', delay=True)
def generate_chat(prompt, model='gpt-3.5-turbo-0125', temperature=1):
    return client.chat.completions.create(model=model, temperature=temperature, messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
    ]).choices[0].message.content



class Config:
    
    def __init__(self, task, **kwargs):
        self.task = task
        self.time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        for k, v in kwargs.items():
            setattr(self, k, v)
        if hasattr(self, "model"):
            self.anthropic_model= 'claude' in self.model
            
    def __str__(self):
        base_str = self.time + "-" + self.task + "-" + self.model
        for k, v in sorted(self.__dict__.items()):
            if k == "time" or k == "task" or k == "model" or k == "bias_text":
                continue
            base_str = base_str + "-" + k.replace("_", "") + str(v).replace("-", "").replace('.json','')
        return base_str


