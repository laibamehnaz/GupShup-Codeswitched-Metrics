class Vocab:
    def __init__(self):
        super().__init__()
        self.size = 0
        self.dict = {}

    @classmethod
    def from_ref_vocab(cls, ref_vocab):
        x = cls()
        x.ref_vocab = ref_vocab
        x.size = 0
        x.index_ref_list = []
        return x

    def add_ref_index(self):
        pass

    def add_word(self, token):
        if token in self.dict:
            self.dict[token] += 1
        else:
            self.dict[token] = 1

    def count(self):
        return len(self.dict.items())

    def display(self, n=5):
        for i in range(min(n, len(self.dict.items()))):
            print(list(self.dict.items())[i])


class Sentence:
    def __init__(self):
        super().__init__()
        self.size = 0
        self.sents = None
        self.actual_ids = []
        self.ref_sents = None
        self.index_ref_sents = []

    @classmethod
    def from_ref_sent(cls, ref_sents):
        x = cls()
        x.ref_sents = ref_sents
        return x

    def add_sent(self, sent, actual_conv_id):
        if self.sents is None:
            self.sents = []
        self.sents.append([sent])
        self.actual_ids.append(actual_conv_id)
        self.size += 1

    def add_ref_index(self, idx, actual_conv_id):  # idx = (sentence,length)
        if type(idx) == list:
            self.index_ref_sents += idx
            self.actual_ids += actual_conv_id
            self.size += len(idx)

        else:
            self.index_ref_sents.append(idx)
            self.actual_ids.append(actual_conv_id)
            self.size += 1

    def count(self):
        return self.size

    def display(self, n=3):
        if self.sents is not None:
            for i in range(min(n, self.size)):
                print(self.sents[i])
        elif self.ref_sents is not None:
            for i in range(min(n, self.size)):
                print(self.ref_sents[self.index_ref_sents[i][0]][self.index_ref_sents[i][1]])
        else:
            ModuleNotFoundError


class Utter:
    def __init__(self):
        super().__init__()
        self.size = 0
        self.utter = None
        self.actual_ids = []
        self.ref_utter = None
        self.index_ref_utter = []

    @classmethod
    def from_ref_utter(cls, ref_utter):
        x = cls()
        x.ref_utter = ref_utter
        return x

    def add_utter(self, utterance, actual_conv_id):
        if self.utter is None:
            self.utter = []
        self.utter.append(utterance)
        self.actual_ids.append(actual_conv_id)
        self.size += 1

    def add_ref_index(self, idx, actual_conv_id):
        """
        idx: idx is a tuple (conversation number, utterance number, number of words in the utterance)
        actual_conv_id: conversation id as present in the corpus
        """
        if type(idx) == list:
            self.index_ref_utter += idx
            self.actual_ids += actual_conv_id
            self.size += len(idx)

        else:
            self.index_ref_utter.append(idx)
            self.actual_ids.append(actual_conv_id)
            self.size += 1

    def count(self):
        return self.size

    def display(self, n=3):
        if self.utter is not None:
            for i in range(min(n, self.size)):
                print(self.utter[i])
        elif self.ref_utter is not None:
            for i in range(min(n, self.size)):
                print(self.ref_utter[self.index_ref_utter[i][0]][self.index_ref_utter[i][1]])
        else:
            ModuleNotFoundError
