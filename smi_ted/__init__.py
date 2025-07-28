from .config import SmiTedHFConfig
from .tokenizer import SmiTedHFTokenizer
from .model import MaterialsSmiTedHFModel
from transformers import AutoConfig, AutoModel, AutoTokenizer

AutoConfig.register("materials.smi-ted", SmiTedHFConfig)
AutoModel.register(SmiTedHFConfig, MaterialsSmiTedHFModel)
AutoTokenizer.register(SmiTedHFConfig, SmiTedHFTokenizer)