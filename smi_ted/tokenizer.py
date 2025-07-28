import re
from transformers import BertTokenizer

class SmiTedHFTokenizer(BertTokenizer):
    vocab_files_names = {"vocab_file": "vocab.txt"}

    def __init__(self, 
                 vocab_file,
                 unk_token='<pad>',
                 sep_token='<eos>',
                 pad_token='<pad>',
                 cls_token='<bos>',
                 mask_token='<mask>',
                 **kwargs):
        super().__init__(vocab_file,
                         unk_token=unk_token,
                         sep_token=sep_token,
                         pad_token=pad_token,
                         cls_token=cls_token,
                         mask_token=mask_token,
                         **kwargs)
        
        PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex_tokenizer = re.compile(PATTERN)
        self.wordpiece_tokenizer = None
        self.basic_tokenizer = None
        with open(vocab_file) as f:
            self.padding_idx = f.readlines().index(pad_token+'\n')

    def _tokenize(self, text):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens

    def get_padding_idx(self):
        return self.padding_idx

    def convert_idx_to_tokens(self, idx_tensor):
        tokens = [self.convert_ids_to_tokens(idx) for idx in idx_tensor.tolist()]
        return tokens
    
    def convert_tokens_to_string(self, tokens):
        stopwords = ['<bos>', '<eos>', '<pad>']
        return ''.join([word for word in tokens if word not in stopwords])

    def idx_to_smiles(self, idx):
        """
        Convert token indices back to SMILES string.
        idx: 1D or 2D list/tensor of token indices.
        """
        rev_tokens = self.convert_idx_to_tokens(idx)
        # Flatten if 2D
        if isinstance(rev_tokens[0], list):
            rev_tokens = [item for sublist in rev_tokens for item in sublist]
        return self.convert_tokens_to_string(rev_tokens)