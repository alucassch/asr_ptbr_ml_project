import glob
import zipfile
from pathlib import Path
from os.path import abspath
from collections import defaultdict

import sentencepiece as spm

from .helper import tmpdir

from typing import Union, DefaultDict

def get_text_token_count(text_file: Union[Path, str], bpe_model: Union[Path, str] = None) -> DefaultDict[str, int]:
    """Extrai a contagem de tokens char ou BPE do arquivo das transcrições usado no treinamento

    Args:
        text_file (Union[Path, str]): arquivo dos textos das transcrições do dataset de treinamento
        bpe_model (Union[Path, str], optional): Arquivo do modelo BPE ou zipfile final do modelo contendo o modelo BPE. Defaults to None.

    Raises:
        ValueError: Caso não consiga carregar o modelo BPE

    Returns:
        DefaultDict[str, int]: Contagem de todos os itens do texto por caracter ou por token BPE
    """
    text_file =  abspath(text_file)
    c: DefaultDict[str, int] = defaultdict(int)
    
    if bpe_model:
        sp = spm.SentencePieceProcessor()
        bpe_model = abspath(bpe_model)

        if zipfile.is_zipfile(bpe_model):
            with tmpdir() as t:
                zipfile.ZipFile(bpe_model).extractall(path=t)
                tmp_bpe_model_file = glob.glob(t+'/data/token_list/bpe*/*')
                if tmp_bpe_model_file:
                    tmp_bpe_model_file = tmp_bpe_model_file[0]
                else:
                    raise ValueError(f"BPE model não encontrado dentro do modelo {bpe_model}")

                sp.load(tmp_bpe_model_file)
        else:
            sp = sp.load(bpe_model)
        
        with open(text_file) as f:
            for line in f:
                text = ' '.join(line.strip().split()[1:])
                token_list = sp.EncodeAsPieces(text)
                for token in token_list:
                    c[token]+=1
    else:
        with open(text_file) as f:
            for line in f:
                char_list = list(line.strip().split()[1:][0])
                for char in char_list:
                    c[char]+=1
        
    return c
    