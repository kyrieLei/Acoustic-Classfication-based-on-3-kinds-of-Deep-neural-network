class config:
    def __init__(self):
        self.sample_rate = 48000
        self.n_fft = 1024
        self.win_length = self.n_fft
        self.hop_length = int(self.win_length / 4)
        self.n_mels = 256

        self.process_data = True
        self.process_data_f = 'new_data/'

        self.epochs = 100
        self.batch_size = 32
        #augmentation
        self.DIFF_FREQ = False
        self.MIXUP= False
        self.SPEC_AUG= True
        #Mixing rate of class
        self.mix_rate=0.8