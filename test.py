from llm_sdk import Small_LLM_Model
from src.vocab import Vocabulary
from pprint import pprint


model = Small_LLM_Model()

vocab = Vocabulary(model.get_path_to_vocab_file())

pprint(vocab.items())