from .load import Smi_ted
from .config import SmiTedHFConfig
from transformers import PreTrainedModel

class MaterialsSmiTedHFModel(PreTrainedModel):
    config_class = SmiTedHFConfig

    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.smi_ted = Smi_ted(config=config.to_dict(), tokenizer=tokenizer)

    def forward(self, *args, **kwargs):
        return self.smi_ted(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.smi_ted.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.smi_ted.decode(*args, **kwargs)