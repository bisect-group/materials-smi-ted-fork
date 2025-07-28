from transformers import PretrainedConfig

class SmiTedHFConfig(PretrainedConfig):
    model_type = "materials.smi-ted"

    def __init__(
        self,
        n_layer=12,
        n_head=12,
        n_embd=768,
        max_len=202,
        d_dropout=0.1,
        num_feats=32,
        smi_ted_version="v1",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.max_len = max_len
        self.d_dropout = d_dropout
        self.num_feats = num_feats
        self.smi_ted_version = smi_ted_version