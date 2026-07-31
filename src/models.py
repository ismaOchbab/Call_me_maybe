from typing import Dict
from pydantic import BaseModel, model_validator


class FunctionSchemaError(Exception):
    """Function schema error"""
    pass


class FunctionSchema(BaseModel):
    """
    Function schema
    """
    name: str
    description: str
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str]

    @model_validator(mode="after")
    def validate_func(self) -> "FunctionSchema":
        """validates data after init"""
        for key in self.parameters.keys():
            if 'type' not in self.parameters[key].keys():
                raise FunctionSchemaError(
                    f"Unsupported argument type for parameter '{key}'"
                    f" in function '{self.name}'"
                )
            if self.parameters[key]['type'] not in ('number',
                                                    'string',
                                                    'boolean'):
                raise FunctionSchemaError(
                                    f"Unsupported argument type for parameter"
                                    f" '{key}'"
                                    f" in function '{self.name}'"
                                )

        return self


class Prompt(BaseModel):
    """
    Single natural language prompt to translate into func call
    """
    prompt: str