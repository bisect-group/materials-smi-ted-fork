# SMILES-based Transformer Encoder-Decoder (SMI-TED)

This repository is a fork of the original source code and adapted for usage with HuggingFace AutoModel.

**Forked GitHub:** [GitHub Link](https://github.com/bisect-group/materials-smi-ted-fork)

**Forked HuggingFace:** [HuggingFace Link](https://huggingface.co/bisectgroup/materials-smi-ted-fork)

The original repository provides PyTorch source code associated with the publication, "A Large Encoder-Decoder Family of Foundation Models for Chemical Language".

**Paper:** [Arxiv Link](https://arxiv.org/abs/2407.20267)

**Original GitHub:** [GitHub Link](https://github.com/IBM/materials/tree/main/models/smi_ted)

**Original HuggingFace:** [HuggingFace Link](https://huggingface.co/ibm/materials.smi-ted)

For more information contact: eduardo.soares@ibm.com or evital@br.ibm.com.


## Usage

```bash
pip install smi-ted
```

```python
import torch
import smi_ted
from transformers import AutoConfig, AutoModel, AutoTokenizer

config = AutoConfig.from_pretrained("bisectgroup/materials-smi-ted-fork")
tokenizer = AutoTokenizer.from_pretrained("bisectgroup/materials-smi-ted-fork")
model = AutoModel.from_pretrained("bisectgroup/materials-smi-ted-fork")

model.smi_ted.tokenizer = tokenizer
model.smi_ted.set_padding_idx_from_tokenizer()

smiles = ['CC1C2CCC(C2)C1CN(CCO)C(=O)c1ccc(Cl)cc1',
'COc1ccc(-c2cc(=O)c3c(O)c(OC)c(OC)cc3o2)cc1O',
'CCOC(=O)c1ncn2c1CN(C)C(=O)c1cc(F)ccc1-2',
'Clc1ccccc1-c1nc(-c2ccncc2)no1',
'CC(C)(Oc1ccc(Cl)cc1)C(=O)OCc1cccc(CO)n1']

with torch.no_grad():
    encoder_outputs = model.encode(smiles)
with torch.no_grad():
    decoded_smiles = model.decode(encoder_outputs)
```

If you use this model, please cite:

```bibtex
@article{soares2025open,
  title={An open-source family of large encoder-decoder foundation models for chemistry},
  author={Soares, Eduardo and Vital Brazil, Emilio and Shirasuna, Victor and Zubarev, Dmitry and Cerqueira, Renato and Schmidt, Kristin},
  journal={Communications Chemistry},
  volume={8},
  number={1},
  pages={193},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
@article{soares2024large,
  title={A large encoder-decoder family of foundation models for chemical language},
  author={Soares, Eduardo and Shirasuna, Victor and Brazil, Emilio Vital and Cerqueira, Renato and Zubarev, Dmitry and Schmidt, Kristin},
  journal={arXiv preprint arXiv:2407.20267},
  year={2024}
}
```