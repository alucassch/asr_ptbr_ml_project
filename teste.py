

if __name__ == "__main__":
    from ASRDecoder.asr import ASR

    #zip criado pelo script de treinamento, deve estar no mesmo diretorio que o notebook
    model_file = 'asr_train_commonvoice_vggrnn_raw_char_valid.acc.ave.zip'
    #arquivo de audio para testes deve ser 16kHz
    audio_file = 'teste_pesquisa.wav'

    #carrega o modelo
    asr = ASR(model_file)

    results = asr.recognize(audio_file)