import copy
from IPython.lib.display import Audio
import yaml
import torch
import shutil
import zipfile
import librosa
import numpy as np

from pathlib import Path
from os.path import abspath, join, exists, dirname
from typing import Tuple, Any, Dict, Union, TypeVar

from espnet2.tasks.asr import ASRTask
from espnet.nets.beam_search import BeamSearch
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet2.torch_utils.device_funcs import to_device
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions

from .result import Result

T = TypeVar('T', bound='Model')

class ASR(object):
    def __init__(
        self, 
        zip_model_file: Union[Path, str],
    ) -> None:
        
        self.zip_model_file = abspath(zip_model_file)
        self.device = 'cpu'
        self.model_config = self.extract_zip_model_file(self.zip_model_file)
        self.build_beam_search()
        self.build_tokenizer()

    def build_beam_search(self, ctc_weight: float = 0.4, beam_size: int = 1):
        """Constroi o objeto beam_search.

        Esse objeto faz a decodificação do vetor de embeddings da saída da parte encoder
        do modelo passando pelos decoders da rede que são o módulo CTC e Transformer ou RNN.

        Como:
        Loss = (1-λ)*DecoderLoss + λ*CTCLoss 
        Se ctc_weight=1 apenas o módulo CTC será usado na decodificação

        Args:
            ctc_weight (float, optional): Peso dado ao módulo CTC da rede. Defaults to 0.4.
            beam_size (int, optional): Tamanho do feixe de busca durante a codificação. Defaults to 1.
        """
        scorers = {}
        ctc = CTCPrefixScorer(ctc=self.model.ctc, eos=self.model.eos)
        token_list = self.model.token_list
        scorers.update(
            decoder=self.model.decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        #Variáveis com os pesos para cada parte da decodificação
        #lm referente à modelos de linguagem não são utilizados aqui mas são necessários no objeto
        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=1.0,
            length_bonus=0.0,
        )

        #Cria o objeto beam_search
        self.beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=self.model.sos,
            eos=self.model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )

        self.beam_search.to(device=self.device, dtype=getattr(torch, 'float32')).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=self.device, dtype=getattr(torch, 'float32')).eval()

    def build_tokenizer(self):
        """Cria um objeto tokenizer para conversão dos tokens inteiros para o dicionário
        de caracteres correspondente.

        Caso o modelo possua um modelo BPE de tokenização, ele é utilizado. Se não, apenas a lista
        de caracteres no arquivo de configuração é usada.
        """
        token_type = self.model_config['token_type']
        if token_type == 'bpe':
            bpemodel = self.model_config['bpemodel']
            self.tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
        else:
            self.tokenizer = build_tokenizer(token_type=token_type)
        
        self.converter = TokenIDConverter(token_list=self.model.token_list)


    def get_layers(self) -> Dict[str, Dict[str, torch.Size]]:
        """Retorna as camadas nomeadas e os respectivos shapes para todos os módulos da rede.

        Os módulos são:
            Encoder: RNN, VGGRNN, TransformerEncoder
            Decoder: RNN, TransformerDecoder
            CTC

        Returns:
            Dict[str, Dict[str, torch.Size]]: Dicionário de cada módulo com seus respectivos layers e shape
        """
        r = {}

        r['encoder'] = {x: self.model.encoder.state_dict()[x].shape 
                            for x in self.model.encoder.state_dict().keys()}
        r['decoder'] = {x: self.model.decoder.state_dict()[x].shape 
                            for x in self.model.decoder.state_dict().keys()}
        r['ctc']     = {x: self.model.ctc.state_dict()[x].shape 
                            for x in self.model.ctc.state_dict().keys()}
        return r


    def frontend(self, audiofile: Union[Path, str], normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executa o frontend do modelo, transformando as amostras de áudio em parâmetros log mel spectrogram

        Args:
            audiofile (Union[Path, str]): arquivo de áudio

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Parâmetros, Tamanho do vetor de parâmetros
        """
        audio_samples, _ = librosa.load(audiofile, sr=16000)
        if isinstance(audio_samples, np.ndarray):
            audio_samples = torch.tensor(audio_samples)
        audio_samples = audio_samples.unsqueeze(0).to(getattr(torch, 'float32'))
        lengths = audio_samples.new_full([1], dtype=torch.long, fill_value=audio_samples.size(1))
        features, features_length  = self.model.frontend(audio_samples, lengths)

        if normalize:
            features, features_length = self.model.normalize(features, features_length)

        return features, features_length

    def specaug(self, features: torch.Tensor, features_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executa o módulo specaug, da parte de 'data augmentation'.
        Útil para visualização apenas. 
        Não é utilizado na inferência, apenas no treinamento.

        Args:
            features (torch.Tensor): Parâmetros
            features_length (torch.Tensor): tamanho do vetor de parâmetros

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Parâmetros com máscaras temporais, em frequência e distoção. Tamanho dos vetores
        """
        return self.model.specaug(features, features_length)

    def extract_zip_model_file(self, zip_model_file: str) -> Dict[str, Any]:
        """Extrai os dados de um zip contendo o arquivo com o estado do modelo e configurações

        Args:
            zip_model_file (str): ZipFile do modelo gerado dos scripts de treinamento

        Raises:
            ValueError: Se o arquivo não for correto
            FileNotFoundError: Se o arquivo zip não contiver os arquivos necessários

        Returns:
            Dict[str, Any]: Dicionário do arquivo .yaml utilizado durante o treinamento para carregar o modelo corretamente
        """
        if not zipfile.is_zipfile(zip_model_file):
            raise ValueError(f"File {zip_model_file} is not a zipfile")
        else:
            zipfile.ZipFile(zip_model_file).extractall(dirname(zip_model_file))

        check = ['exp', 'meta.yaml']

        if not all([x for x in check]):
            raise FileNotFoundError
        
        with open('meta.yaml') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)

        model_stats_file = meta['files']['asr_model_file']
        asr_model_config_file = meta['yaml_files']['asr_train_config']
        
        asr_model_config = {}
        with open(asr_model_config_file) as f:
            asr_model_config = yaml.load(f, Loader=yaml.FullLoader)
            try:
                self.modelglobal_cmvn = asr_model_config['normalize_conf']['stats_file']
            except KeyError:
                self.global_cmvn = None

        print(f'Loading model config from {asr_model_config_file}')
        print(f'Loading model state from {model_stats_file}')

        #Build Model
        self.model, _ = ASRTask.build_model_from_file(
            asr_model_config_file, model_stats_file, self.device
        )
        self.model.to(dtype=getattr(torch, 'float32')).eval()

        return asr_model_config

    def __del__(self) -> None:
        """Remove os arquivos temporários
        """
        print(f"Removing exp dir {join(dirname(self.zip_model_file), 'exp')}")
        if exists(join(dirname(self.zip_model_file), 'exp')):
            shutil.rmtree(join(dirname(self.zip_model_file), 'exp'))
        
        print(f"Removing meta.yaml {join(dirname(self.zip_model_file), 'meta.yaml')}")
        if exists(join(dirname(self.zip_model_file), 'exp')):
            shutil.rmtree(join(dirname(self.zip_model_file), 'exp'))

    @torch.no_grad()
    def recognize(self, audiofile: Union[Path, str]) -> Result:

        result = Result()
        
        audio_samples, rate = librosa.load(audiofile, sr=16000)

        result.audio_samples = copy.deepcopy(audio_samples)

        #a entrada do modelo é torch.tensor
        if isinstance(audio_samples, np.ndarray):
            audio_samples = torch.tensor(audio_samples)
        audio_samples = audio_samples.unsqueeze(0).to(getattr(torch, 'float32'))
        
        lengths = audio_samples.new_full([1], dtype=torch.long, fill_value=audio_samples.size(1))
        batch = {"speech": audio_samples, "speech_lengths": lengths}
        batch = to_device(batch, device=self.device)

        #model encoder
        enc, _ = self.model.encode(**batch)

        #model decoder
        nbest_hyps = self.beam_search(x=enc[0])

        #Apenas a melhor hipótese
        best_hyps = nbest_hyps[0]

        #Conversão de tokenids do treinamento para texto
        token_int = best_hyps.yseq[1:-1].tolist()
        token_int = list(filter(lambda x: x != 0, token_int))
        token = self.converter.ids2tokens(token_int)
        text = self.tokenizer.tokens2text(token)

        #Preenche o objeto result
        result.text = text
        result.encoded_vector = enc[0] #[0] remove dimensão de batch
        
        #calcula todas as matrizes de atenção aij = softmax(QK/sqrt(d))*V
        #onde Q, K, V são transformações lineares y=wx+b
        #self attention: Q, K e V são os mesmos vetores
        #source attention: Q, K são o resultado do encoder e V o resultado do decoder a cada passo da decodificação
        text_tensor = torch.Tensor(token_int).unsqueeze(0).to(getattr(torch, 'long'))
        batch["text"] = text_tensor
        batch["text_lengths"] = text_tensor.new_full([1], dtype=torch.long, fill_value=text_tensor.size(1))
        
        result.attention_weights = calculate_all_attentions(self.model, batch)
        result.tokens_txt = token

        #CTC posteriors
        logp = self.model.ctc.log_softmax(enc.unsqueeze(0))[0]
        result.ctc_posteriors = logp.exp_().numpy()
        result.tokens_int = best_hyps.yseq
        result.mel_features, _ = self.frontend(audiofile, normalize=False)
        return result

    def __call__(self, input: Union[Path, str]) -> Result:
        return self.process(input)


