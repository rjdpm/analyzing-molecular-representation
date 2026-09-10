"""
-----------------------------------------------------------------------------
For dataset helpers
-----------------------------------------------------------------------------
AUTHOR: Soumitra Samanta                            DATE: Friday, 05012241030
For bug and others mail me at (soumitra.samanta@gm.rkmvu.ac.in)
-----------------------------------------------------------------------------
"""


from typing import Union, Tuple, Dict, List, Any
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import sys
import os

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer

import selfies as sf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from input_output import *

# # for auto-reloading external modules
# # see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2


__all__ = [
    '_check_validity_smiles',
    '_canonicalise_smiles',
    '_check_validity_selfies_from_smiles',
    '_check_smiles_validity',
    '_check_validity_selfies',
    '_smiles2selfies',
    '_selfies2smiles',
    '_find_max_len_selfies',
    '_get_tokens_selfies',
    '_get_dict_selfies',
    '_selfies2onehot',
    '_onehot2selfies',
    '_read_max_len_selfies',
    '_read_tokens_selfies',
    'descriptors_mol',
]

def _check_validity_smiles(
    smiles: List[str], 
    verbose: bool = True
) ->Tuple[List]:
    """
    Check smiles is valid or not!

    INPUT: 
    smiles- input smiles as a "list"
    verbose- display progress bar during checking

    OUTPUT:
    smiles- valid smiles (only)
    check_ids- smiles validity flag (True or False) of the input smiles
    """

    if verbose:
        print('='*70)
        print('<===checking smiles validity===>')
        print('-'*70)
    if verbose:
        iterator = enumerate(tqdm(smiles))
    else:
        iterator = enumerate(smiles)

    check_ids = [True]*len(smiles)
    for i, sm in iterator:
        try:
            m = Chem.MolFromSmiles(sm)
            if m is None:
                check_ids[i] = False 
            else:  
                try:
                    Chem.Kekulize(m)
                except:
                    check_ids[i] = False 
        except:
            check_ids[i] = False
    smiles = [smiles[i] for i, flag in enumerate(check_ids) if flag]
    if verbose:
        print('='*70)

    return smiles, check_ids


def _canonicalise_smiles(
    smiles: List[str], 
    verbose: bool = True
) -> List[str]:
    """
    Canonicalise all smiles
    
    INPUT: 
    smiles- input smiles as a "list"
    verbose- display progress bar during checking

    OUTPUT:
    smiles- canonical smiles
    """

    if verbose:
        print('='*70)
        print('<===canonicalising smiles===>')
        print('-'*70)
    if verbose:        
        iterator = enumerate(tqdm(smiles))
    else:
        iterator = enumerate(smiles)
    
    return [Chem.MolToSmiles(Chem.MolFromSmiles(sm), isomericSmiles=False) for i, sm in iterator]


def _check_validity_selfies_from_smiles(
    smiles: List[str], 
    flag_decode_smiles: bool = True,
    verbose: bool = True
) ->Tuple[List]:
    """
    Check smiles has a valid selfies or not!

    INPUT: 
    smiles- input smiles as a "list"
    flag_decode_smiles- need actual decoded smiles from the felfies or not? 
    verbose- display progress bar during checking

    OUTPUT:
    selfiles- valid selfies (only)
    smiles- valid smiles corresponding to the valid selfies (only)
    check_ids- selfies validity flag (True or False) of the input smiles
    smiles_decode- decoded smiles from selfies
    """

    if verbose:
        print('='*70)
        print('<===checking selfies validity===>')
        print('-'*70)
        
    selfiles = [None]*len(smiles)
    if flag_decode_smiles:
        smiles_decode = [None]*len(smiles)
    else:
        smiles_decode = []
    check_ids = [True]*len(smiles)
    
    if verbose:
        iterator = enumerate(tqdm(smiles))
    else:
        iterator = enumerate(smiles)
    for i, sm in iterator:
        try:
            selfiles[i] = sf.encoder(sm)
            temp_sm = sf.decoder(selfiles[i])  
            if flag_decode_smiles:
                smiles_decode[i] = temp_sm
            # selfies might have decode different smiles, os need to check the canonical form of decode smiles
            m = Chem.MolFromSmiles(temp_sm)
            if m is None: 
                check_ids[i] = False
            else:
                if sm != Chem.MolToSmiles(m, isomericSmiles=False):
                    check_ids[i] = False
        except sf.EncoderError:
            check_ids[i] = False   # sf.encoder error!
        except sf.DecoderError:
            check_ids[i] = False   # sf.decoder error!

            
    selfiles = [selfiles[i] for i, flag in enumerate(check_ids) if flag]
    smiles = [smiles[i] for i, flag in enumerate(check_ids) if flag]
    if verbose:
        print('='*70)

    return selfiles, smiles, check_ids, smiles_decode

def _check_smiles_validity(smiles, verbose=1):
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

def _check_validity_selfies(
    selfiles: List[str], 
    flag_decode_smiles: bool = True,
    verbose: bool = True
) ->Tuple[List]:
    """
    Check selfiles has a valid smiles or not!

    INPUT: 
    selfiles- input selfiles as a "list"
    flag_decode_smiles- need actual decoded smiles from the felfies or not? 
    verbose- display progress bar during checking

    OUTPUT:
    selfiles- valid selfies (only)
    check_ids- selfies validity flag (True or False) of the input smiles
    smiles_decode- decoded smiles from selfies
    """

    if verbose:
        print('='*70)
        print('<===checking selfies validity===>')
        print('-'*70)
        
    if flag_decode_smiles:
        smiles_decode = [None]*len(selfiles)
    else:
        smiles_decode = []
    check_ids = [True]*len(selfiles)
    
    if verbose:
        iterator = enumerate(tqdm(selfiles))
    else:
        iterator = enumerate(selfiles)
    for i, s in iterator:
        try:
            temp_sm = sf.decoder(s)  
            if flag_decode_smiles:
                smiles_decode[i] = temp_sm
            # selfies might have decode different smiles, os need to check the canonical form of decode smiles
            m = Chem.MolFromSmiles(temp_sm)
            if m is None: 
                check_ids[i] = False
            else:
                if temp_sm != Chem.MolToSmiles(m, isomericSmiles=False):
                    check_ids[i] = False
            _, temp_check_ids = _check_validity_smiles(
                [temp_sm],
                verbose=False
            )
            check_ids[i] = temp_check_ids[0]
            if check_ids[i]:# conver smiles to cannonical smiles 
                smiles_decode[i] = _canonicalise_smiles(
                    [temp_sm],
                    verbose=False
                )[0]
        except sf.DecoderError:
            check_ids[i] = False   # sf.decoder error!

            
    selfiles = [selfiles[i] for i, flag in enumerate(check_ids) if flag]
    if verbose:
        print('='*70)

    return selfiles, check_ids, smiles_decode


def _smiles2selfies(
    smiles: List[str], 
    flag_valid_trans: bool = True,
    verbose: bool = True
) ->Tuple[List]:
    """
    Transform smiles to selfies

    INPUT: 
    smiles- input smiles as a "list"
    flag_valid_trans- check transformation is valid or not 
    verbose- display progress bar during checking

    OUTPUT:
    selfiles- valid selfies (only)
    check_ids- selfies validity flag (True or False) of the input smiles
    """

    if verbose:
        print('='*70)
        print('<===transforming smiles to selfies===>')
        print('-'*70)
        
    selfiles = [None]*len(smiles)
    if flag_valid_trans:
        check_ids = [True]*len(smiles)
    else:
        check_ids = []
        
    if verbose:
        iterator = enumerate(tqdm(smiles))
    else:
        iterator = enumerate(smiles)
    for i, sm in iterator:
        try:
            selfiles[i] = sf.encoder(sm)
        except sf.EncoderError:
            if flag_valid_trans:
                check_ids[i] = False   # sf.encoder error!
    if verbose:
        print('='*70)

    return selfiles, check_ids

def _selfies2smiles(
    selfies: List[str],
    flag_valid_trans: bool = True,
    verbose: bool = True
) -> Tuple[List]:
    """
    Transform selfies to smiles
    
    INPUT: 
    selfies- input selfies as a "list"
    flag_valid_trans- check transformation is valid or not 
    verbose- display progress bar during checking

    OUTPUT:
    smiles- smiles
    check_ids- valid transformatiion checks
    """
    
    if verbose:
        print('='*70)
        print('<===transforming selfies to smiles===>')
        print('-'*70)
        
    smiles = [None]*len(selfies)
    if flag_valid_trans:
        check_ids = [True]*len(selfies)
    else:
        check_ids = []
        
    if verbose:
        iterator = enumerate(tqdm(selfies))
    else:
        iterator = enumerate(selfies)
    for i, s in iterator:
        try:
            smiles[i] = sf.decoder(s)  
        except sf.DecoderError:
            if flag_valid_trans:
                check_ids[i] = False   # sf.decoder error!
    if verbose:
        print('='*70)

    return smiles, check_ids


def _find_max_len_selfies(
    selfies: List[str],
    save_filename: str = 'max_len_selfies.txt',
    verbose: bool = True
) -> int:
    """
    Find the maximum selfies length
    
    INPUT: 
    selfies- input selfies as a "list"
    save_filename- file name (.txt) to save the maximum length 
    verbose- display progress bar during checking

    OUTPUT:
    max_len_selfies- maximum selfies length
    """
    
    if verbose:
        print('='*70)
        print('<===Finding maximum selfies length===>')
        print('-'*70)
    
    max_len_selfies= 0
    
    if verbose:
        iterator = enumerate(tqdm(selfies))
    else:
        iterator = enumerate(selfies)
    for i, s in iterator:
        len_selfies = sf.len_selfies(s)
        if len_selfies > max_len_selfies:
            max_len_selfies = len_selfies
    save_filename = ''.join([
        save_filename[:-4],
        '_', str(max_len_selfies),
        '.txt'
    ])
    save_value_to_txt(max_len_selfies, save_filename)
    if verbose:
        print('='*70)
    
    return max_len_selfies
    
    
def _get_tokens_selfies(
    selfies: List[str],
    start_token: str = 'sos',#start-of-string(sos)
    end_token: str = 'eos',#end-of-string(sos)
    save_filename: str = 'token_selfies.csv',
    verbose: bool = True
) -> List:
    """
    Get all the tokens from a list pf selfies and saved as a csv file
    
    INPUT: 
    selfies- input selfies as a "list"
    start_token- start token for the selfies string 
    end_token- end token for the selfies string 
    save_filename- file name (.csv) to save the tokens
    verbose- display progress bar during checking

    OUTPUT:
    tokens- selfies tokens
    """
    
    if verbose:
        print('='*70)
        print('<===getting selfies tokens===>')
        print('-'*70)
    
    tokens = []
        
    if verbose:
        tokens = sf.get_alphabet_from_selfies(tqdm(selfies))
    else:
        tokens = sf.get_alphabet_from_selfies(selfies)
    tokens = list(tokens)
    
    #add start and end token
    if start_token not in tokens:
        tokens.append(start_token)
    else:
        raise ValueError('Start token "{}" is aleardy in the selfies token list!\nPlease change the start token?' .format(start_token))
    if end_token not in tokens:
        tokens.append(end_token)
    else:
        raise ValueError('End token "{}" is aleardy in the selfies token list!\nPlease change the end token?' .format(end_token))
    
    tokens = sorted(tokens)
    
    # save tokens as .csv file
    save_filename = ''.join([
        save_filename[:-4],
        '_', str(len(tokens)),
        '.csv'
    ])
    save_dict_csv_pandas(
        {'tokens': tokens},
        save_filename
    )
    if verbose:
        print('='*70)

    return tokens


def _get_dict_selfies(
    tokens: List[str],
) ->Dict:
    """
    Create dictionary for selfies tokens
    
    INPUT: 
    tokens- selfies tokens

    OUTPUT:
    dict_selfies- selfies token dictionar for doamin and codoamin
    """
    
    dict_selfies = OrderedDict({
        'domain': OrderedDict({}),
        'codomain': OrderedDict({})
    })
    
    tokens = sorted(tokens)
    for i, s in enumerate(tokens):
        dict_selfies['domain'][s] = i
        dict_selfies['codomain'][i] = s
        
    return dict_selfies


def _selfies2onehot(
    selfies: List[str],
    max_selfies_len: int,
    dict_selfies: Dict,
    padded_token: str = 'eos',#end-of-string(sos)
    flag_interget_vector: bool = False,
    verbose: bool = True
) -> Tuple[np.asarray]:
    """
    Transform selfies to one-hot-vector
    
    INPUT: 
    selfies- given selfies as alist
    max_selfies_len- mamimum selfies string length
    dict_selfies- selfies token dictionary
    padded_token- token to fill extra space at the end
    flag_interget_vector- need integer vector
    verbose- display progress bar during checking

    OUTPUT:
    onehot_vectors- one-hot represenattion of the selfies
    interger_vectors- inter vector representation of the selfies
    """
    
    if verbose:
        print('='*70)
        print('<===getting one-hot vector representation===>')
        print('-'*70)
    
    #check padding token in the dictionary
    if padded_token not in dict_selfies['domain'].keys():
        raise ValueError('Padding token "{}" is not in the selfies token list!\nPlease check the tokens dictionary?' .format(padded_token))
        
    onehot_vectors = np.zeros((len(selfies), max_selfies_len, len(dict_selfies['domain'].keys())), dtype=np.bool_)
    if flag_interget_vector:
        interger_vectors = np.zeros((len(selfies), max_selfies_len))
    else:
        interger_vectors = np.asarray([])
    
    if verbose:
        iterator = enumerate(tqdm(selfies))
    else:
        iterator = enumerate(selfies)
        
    for i, s in iterator:
        tokens = list(sf.split_selfies(s))
        tokens += [padded_token] * (max_selfies_len - len(tokens))
        for j, t in enumerate(tokens):
            try:
                onehot_vectors[i, j, dict_selfies['domain'][t]] = 1
                if flag_interget_vector:
                    interger_vectors[i, j] = dict_selfies['domain'][t]
            except KeyError as e:
                print("ERROR: Check token file. bad SELFIES:", s)
                raise e
    if verbose:
        print('='*70)
    
    return onehot_vectors, interger_vectors

    
def _onehot2selfies(
    X: np.asarray,
    indices_tokens: Dict,
    padded_token: str = 'eos',#end-of-string(sos)
    remove_padded_token: bool = True,
    verbose: bool = True
) -> List:
    """
    Transform one-hot-vector to selfies
    
    INPUT: 
    X- given one-hot vectors as a numpy matrix
    dict_selfies- selfies token dictionary
    padded_token- token to fill extra space at the end
    remove_padded_token- remove padded token (end of string) or not
    verbose- display progress bar during checking

    OUTPUT:
    selfies- selfies representation of the one-hot vectors
    """
    
    if verbose:
        print('='*70)
        print('<===transforming one-hot vectors to selfies===>')
        print('-'*70)
        
    selfies = [None]*X.shape[0]
    
    if verbose:
        iterator = tqdm(range(X.shape[0]))
    else:
        iterator = range(X.shape[0])
        
    for i in iterator:
        selfies[i] = ""
        index = np.argmax(X[i,:,:], axis=1)
        for j in index:
            selfies[i] = ''.join([selfies[i], indices_tokens[j]])
        if remove_padded_token:
            selfies[i] = selfies[i].replace(padded_token, "")
        selfies[i] = selfies[i].strip()
    if verbose:
        print('='*70)
    
    return selfies
        
    
def _read_max_len_selfies(
    input_filename: str
) -> int:
    """
    Read the maximum selfies length from a .txt file
    
    INPUT:
    input_filename- selfies string length input file name (.txt)

    OUTPUT:
    max_len_selfies- selfies string max-length
    """

    with open(input_filename, 'r') as f:
        max_len_selfies = int(f.readline())

    return max_len_selfies


def _read_tokens_selfies(
    input_filename: str
) -> List:
    """
    Read selfies tokens from a .csv file
    
    INPUT:
    input_filename- selfies tokens input file name (.csv)

    OUTPUT:
    tokens- selfies tokens
    """

    dict_tokens = read_csv_file_to_dict(input_filename)

    return dict_tokens['tokens']


class descriptors_mol():
    """Molecular descriptors from SMILES string"""
    
    def __init__(
        self, 
        smiles: List[str]
    ) -> None:
        self.smiles = smiles
        
        
    def _logp(
        self,
        verbose=True
    )->List:
        """
        logP- Partition Coefficient: log10 (P); P = Partition Coefficient = [organic]/[aqueous]
        Where [ ] indicates the concentration of solute in the organic and aqueous partition
        """

        if verbose:
            print('='*70)
            print('<===Calculating "LogP" values===>')
            print('-'*70)
    
        descriptor_val = [None]*len(self.smiles)
        if verbose:
            iterator = enumerate(tqdm(self.smiles))
        else:
            iterator = enumerate(self.smiles)
        for i, sm in iterator:
            descriptor_val[i] = Descriptors.MolLogP(Chem.MolFromSmiles(sm))
        if verbose:
            print('='*70)

        return descriptor_val
    #------------------------------------------------------
    
    
    def _qed(
        self,
        verbose=True
    )->List:
        """
        Quantitative estimation of drug-likeness (qed)
        """

        if verbose:
            print('='*70)
            print('<===Calculating "QED" values===>')
            print('-'*70)
    
        descriptor_val = [None]*len(self.smiles)
        if verbose:
            iterator = enumerate(tqdm(self.smiles))
        else:
            iterator = enumerate(self.smiles)
        for i, sm in iterator:
            descriptor_val[i] = Descriptors.qed(Chem.MolFromSmiles(sm))
        if verbose:
            print('='*70)

        return descriptor_val
    #------------------------------------------------------
    
    
    def _sas(
        self,
        verbose=True
    )->List:
        """
        Synthetic assessibility score (SAS)
        """

        if verbose:
            print('='*70)
            print('<===Calculating "SAS" values===>')
            print('-'*70)
    
        descriptor_val = [None]*len(self.smiles)
        if verbose:
            iterator = enumerate(tqdm(self.smiles))
        else:
            iterator = enumerate(self.smiles)
        for i, sm in iterator:
            descriptor_val[i] = sascorer.calculateScore(Chem.MolFromSmiles(sm))
        if verbose:
            print('='*70)

        return descriptor_val
    #------------------------------------------------------

    
    def _sas(
        self,
        verbose=True
    )->List:
        """
        Synthetic assessibility score (SAS)
        """

        if verbose:
            print('='*70)
            print('<===Calculating "SAS" values===>')
            print('-'*70)
    
        descriptor_val = [None]*len(self.smiles)
        if verbose:
            iterator = enumerate(tqdm(self.smiles))
        else:
            iterator = enumerate(self.smiles)
        for i, sm in iterator:
            descriptor_val[i] = sascorer.calculateScore(Chem.MolFromSmiles(sm))
        if verbose:
            print('='*70)

        return descriptor_val
    #------------------------------------------------------
    
    
    def _weight(
        self,
        verbose=True
    )->List:
        """
        Molecular weight (weight)
        """

        if verbose:
            print('='*70)
            print('<===Calculating "molecular weight" ===>')
            print('-'*70)
    
        descriptor_val = [None]*len(self.smiles)
        if verbose:
            iterator = enumerate(tqdm(self.smiles))
        else:
            iterator = enumerate(self.smiles)
        for i, sm in iterator:
            descriptor_val[i] = Descriptors.ExactMolWt(Chem.MolFromSmiles(sm))
        if verbose:
            print('='*70)

        return descriptor_val
    #------------------------------------------------------

    
    def _formula(
        self,
        verbose=True
    )->List:
        """
        Molecular formula
        """

        if verbose:
            print('='*70)
            print('<===Calculating "molecular formula" ===>')
            print('-'*70)
    
        descriptor_val = [None]*len(self.smiles)
        if verbose:
            iterator = enumerate(tqdm(self.smiles))
        else:
            iterator = enumerate(self.smiles)
        for i, sm in iterator:
            descriptor_val[i] = CalcMolFormula(Chem.MolFromSmiles(sm))
        if verbose:
            print('='*70)

        return descriptor_val
    #------------------------------------------------------
    

if __name__ == '__main__':
    
    input_filename = '/Users/soumitra/Documents/SS_DATASET/SS_SUSCORD/SS_MOLECULES/05012024/zinc/iva/data/LogP-1_zinc_smiles_selfies_frag.parquet'
    
    dict_data = read_parquet_file_to_dict_pandas(input_filename)
    print('#SMILES: {}' .format(dict_data['smiles']))
    
    smiles, check_ids = _check_validity_smiles(dict_data['smiles'])
    canon_smiles = _canonicalise_smiles(smiles)
    selfies, canon_smiles_selfies, check_ids, smiles_decode = _check_validity_selfies_from_smiles(canon_smiles)
    
    smiles_2_selfies, check_ids = _smiles2selfies(smiles)
    selfiles_2_smiles, check_ids = _selfies2smiles(selfies)
    
    max_len_selfies = _find_max_len_selfies(smiles_2_selfies)
    tokens = _get_tokens_selfies(smiles_2_selfies)
    dict_selfies = _get_dict_selfies(tokens)
    
    input_filename_max_len_selfies = ''.join(['max_len_selfies_', str(max_len_selfies), '.txt'])
    max_len_selfies = _read_max_len_selfies(input_filename_max_len_selfies)
    
    input_filename_tokens = ''.join(['token_selfies', str(max_len_selfies), '.csv'])
    tokens = _read_tokens_selfies(input_filename_tokens)
    dict_selfies = _get_dict_selfies(tokens)
    
    onehot, _ = _selfies2onehot(smiles_2_selfies, max_len_selfies, dict_selfies)
    
    descriptors = descriptors_mol(canon_smiles)
    logp_val = descriptors._logp()
    qed_val = descriptors._qed()
    sas_val = descriptors._sas()
    mol_weight = descriptors._weight()
    mol_formula = descriptors._formula()
    
    

