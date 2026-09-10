import torch
from rdkit import Chem
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel

__all__ = ['check_smiles_validity',
           'chemberta2_embed']

def check_smiles_validity(smiles, verbose=1):
    """
    Check smiles is valid or not!
    
    INPUT: 
    smiles- input smiles as a "list"
    flag_progress- display progress bar during checking(0- no, >0 display) (default 0)
    
    OUTPUT:
    smiles- valid smiles (only)
    check_ids- smiles validity flag (True or False) of the input smiles
    """
    
    if verbose:
        print('<===checking smiles validity===>')
        iterator = enumerate(tqdm(smiles, total=len(smiles)))
    else:
        iterator = enumerate(smiles)
        
    check_ids = [True]*len(smiles)
    for i, smile in iterator:
        try:
            Chem.Kekulize(Chem.MolFromSmiles(smile))
        except:
            check_ids[i] = False 

    smiles = [smiles[i] for i, flag in enumerate(check_ids) if flag]

    return smiles, check_ids
#------------------------------------------------------

class chemberta2_embed:
    def __init__(self, 
                 name="DeepChem/ChemBERTa-77M-MLM",
                 device='cuda:0' if torch.cuda.is_available() else 'cpu',
                 ):

        self.device = device
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.config = AutoConfig.from_pretrained(name)
        self.config.output_hidden_states = True
        self.config.output_attentions = True
        self.model = AutoModel.from_config(self.config)
        self.model.eval()
        self.model.to(device)

    def smiles2tokens(self, smiles, max_len=None):

        encoded = self.tokenizer(smiles, 
                                 padding=True, 
                                 truncation=True, 
                                 max_length=max_len
                                 )
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        out = {'encoded':encoded, 'tokens':tokens}

        return out

    def tokens2smiles(self, input_ids):
        return self.tokenizer.decode(token_ids=input_ids, skip_special_tokens=True)

    def smiles_list2tokens(self, smiles_list, max_len=128):
        
        if not isinstance(smiles_list, list):
            smiles_list = list(smiles_list)
        tokens = self.tokenizer(smiles_list, 
                                padding="max_length", 
                                truncation=True, 
                                max_length=max_len, 
                                return_tensors="pt"
                                )

        return tokens

    def embed_smiles(self, 
                     smiles_list, 
                     batch_size=256, 
                     pool='mean',
                     max_len=128,
                     ):

        self.model.eval()
        embeddings = []

        for i in range(0, len(smiles_list), batch_size):

            batch = smiles_list[i:i + batch_size]
            tokens = self.smiles_list2tokens(smiles_list=batch, max_len=max_len)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            with torch.no_grad():
                outputs = self.model(input_ids=tokens["input_ids"],
                                attention_mask=tokens["attention_mask"]
                                )
            hidden = outputs.last_hidden_state
            if pool == 'mean':
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            elif pool == 'cls':
                pooled = hidden[:, 0, :]
            else:
                raise ValueError(f'pool should be `mean` or `cls`')
            
            embeddings.append(pooled.cpu())

        return torch.cat(embeddings, dim=0)