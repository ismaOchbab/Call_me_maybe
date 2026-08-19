from typing import Dict
from pydantic import BaseModel, model_validator


class FunctionSchemaError(Exception):
    """Function schema error"""
    pass


class FunctionSchema(BaseModel):
    """
    Function schema to contain the parsed input file
    """
    name: str
    description: str
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str]
    model_config = {"extra": "forbid"}

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
                                                    'integer',
                                                    'string',
                                                    'boolean'):
                raise FunctionSchemaError(
                                    f"Unsupported argument type for parameter"
                                    f" '{key}'"
                                    f" in function '{self.name}'"
                                )
        if 'type' not in self.returns:
            raise FunctionSchemaError(
                f"Missing return type in function '{self.name}' "
            )
        if self.returns['type'] not in ('number',
                                        'integer',
                                        'string',
                                        'boolean'):
            raise FunctionSchemaError(
                f"Unsupported return type in function '{self.name}'"
            )

        return self


class Prompt(BaseModel):
    """
    Single natural language prompt to translate into func call
    """
    prompt: str


class OutputSchema(BaseModel):
    """
    Output layout for every element in the JSON array
    """
    prompt: str
    name: str
    parameters: Dict[str, object]
