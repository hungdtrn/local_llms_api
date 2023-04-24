"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os
import os.path as osp
from typing_extensions import TypedDict
from enum import auto, Enum
from typing import List, Optional, Literal, Union, Iterator, Dict

current_dir = os.path.dirname(__file__)

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join(current_dir, "templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        idx = output.rfind(self.template["response_split"])
        if idx == -1:
            idx = 0
        else:
            idx = idx + len(self.template["response_split"])
            
        return output[idx:].strip(), idx

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    DOLLY = auto()

class Conversation:
    def __init__(self, messages: List[List[str]], system: str="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
                 roles: List[str]=["user", "assistant"], offset: int=0,
                 sep_style: SeparatorStyle = SeparatorStyle.SINGLE,
                 sep: str = " ", sep2: str = "</s>") -> None:
        self.system = system
        self.roles = roles
        self.messages = messages
        self.offset = offset
        self.sep_style = sep_style
        self.sep = sep
        self.sep2 = sep2
    
    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += self.sep + " " + role + ": " + message
                else:
                    ret += self.sep + " " + role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")
       
def turn_to_message_array(messages):
    out = []
    for m in messages:
        out.append([m["role"], m["content"]])
    return out
 