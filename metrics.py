import json
import string
from utils import Vocab, Utter, Sentence
import argparse
import en_core_web_sm

nlp = en_core_web_sm.load()

operator_verbs = set(
    ["hai", "pe", "pene", "peena", "pi", "dena", "rehna", "rehta", "kardiya", "kiye", "puchna", "karein", "dekho",
     "karo", "jao", "kara", "kra", "aayenge", "liyo", "liya", "dekhne", "dekhna", "dekhte", "hogaye", "chahiye", "thi",
     "kiya", "rahi", "raha", "karna", "kar", "liye", "diya", "karlunga", "hogaya", "gya", "gaya", "kaunsi", "chlo",
     "chli", "lagna", "lagi", "ki", "karli", "kardi", "tha", "aayi", "aaya", "aya", "aati", "ati", "ayi"])


class CodeMixedStats:
    def __init__(self, native_lexicon_file, eng_lexicon_file, ner_set_file, data_file, bigrams_file, output_dir):
        super().__init__()
        self.file_path = data_file
        self.output_dir = output_dir
        self.conversation = []
        self.conversation_ids = []
        self.wordbigrams = None

        with open(bigrams_file, encoding='utf-8') as f:
            self.wordbigrams = f.read().split('\n')

        for w in range(len(self.wordbigrams)):
            self.wordbigrams[w] = self.wordbigrams[w].split()

        self.native_lexicon = []  # Words with Hindi tags
        self.eng_lexicon = []  # Words with English tags
        self.ner_set = []  # Words with NER tags

        self.native_lexicon = self.read_tag_file(native_lexicon_file)
        self.eng_lexicon = self.read_tag_file(eng_lexicon_file)
        self.ner_set = self.read_tag_file(ner_set_file, ner_others=True)

        # list of vocab
        self.total_vocab = Vocab()
        self.eng_vocab = Vocab()
        self.native_vocab = Vocab()
        self.other_vocab = Vocab()
        self.ner_vocab = Vocab()
        self.cm_eng_vocab = Vocab()

        # list of utterances
        self.total_utter = Utter().from_ref_utter(self.conversation)
        self.unique_utter = Utter.from_ref_utter(self.conversation)
        self.repeated_utter = Utter.from_ref_utter(self.conversation)
        self.other_utter = Utter.from_ref_utter(self.conversation)
        self.cm_utter = Utter.from_ref_utter(self.conversation)
        self.native_utter = Utter.from_ref_utter(self.conversation)
        self.eng_utter = Utter.from_ref_utter(self.conversation)

        # list of sentences
        self.cm_sents = Sentence()
        self.all_sents = Sentence()
        self.eng_matrix = Sentence.from_ref_sent(self.cm_sents.sents)
        self.hindi_matrix = Sentence.from_ref_sent(self.cm_sents.sents)

        # code mixed statistics:
        self.cm_statistics = {"Cu_avg": 0.0, "Cu_metric": 0.0, "I_index": 0.0}

        # insertions and alternations
        self.eng_insertions = set()
        self.hindi_insertions = set()
        self.eng_alternations = set()
        self.hindi_alternations = set()

    def read_tag_file(self, filepath, ner_others=False):
        file = open(filepath).read().split('\n')
        content = []
        if not ner_others:
            for i in file:
                try:
                    content.append(i.split('\t'))
                except:
                    continue

        elif ner_others:
            for i in file:
                try:
                    content.append(i.split('\t')[1])
                except:
                    continue
        return content

    def is_code_choice(self, token, actual_conv_id):
        """
        Calculates the language tag of the given token.
        Args:
            token: token whose language tag is to be returned.
            actual_conv_id: the conversation id in the Gupshup corpus to which this token belongs to.
        """

        obj = str.maketrans('', '', string.punctuation)
        cleaned_text = str(token.text).translate(obj)

        if cleaned_text in self.ner_set:
            return 'NE'
        for i in self.native_lexicon:
            if str(i[0])[:-2] == str(actual_conv_id) and i[1] == token.text:
                return 'Hindi'
        for i in self.eng_lexicon:
            if str(i[0])[:-2] == str(actual_conv_id) and i[1] == token.text:
                return 'English'

        return 'Other'

    def remove_name(self, sentence):
        """
        Removes the name of the speaker from the given utterance/sentence.
        """
        try:
            index = sentence.index(':')
        except ValueError:
            return sentence
        return sentence[index + 1:]

    def compute_vocabs(self, doc, actual_conv_id, lang):
        """
        Computes all the different vocab sets for the dataset, such as, total_vocab, eng_vocab, ner_vocab,
        cm_eng_vocab, etc.
        Args:
            doc: spaCy doc object
            actual_conv_id: the conversation id in the Gupshup corpus to which this token belongs to.
        """

        for word in doc:
            code_choice = self.is_code_choice(word, actual_conv_id)
            self.total_vocab.add_word(word.text)
            if code_choice == 'English':
                if lang == 'English':
                    self.eng_vocab.add_word(word.text)
                elif lang == 'code_mixed':
                    self.cm_eng_vocab.add_word(word.text)
            elif code_choice == 'Hindi':
                self.native_vocab.add_word(word.text)
            elif code_choice == 'Other':
                self.other_vocab.add_word(word.text)
            elif code_choice == 'NE':
                self.ner_vocab.add_word(word.text)

    def get_total_utterances_and_vocab(self, conv, actual_conv_id):
        """
        Calculates all the different utterance sets and vocabs sets for the dataset,
        for e.g., total_vocab, total_utter, etc.
        Args:
            conv: conversation as read from the dataset
            actual_conv_id : id of the conversation as in the GupShup dataset, eg: 13818513
        """

        dialogues = conv.split("\n")
        self.conversation.append(dialogues)
        self.conversation_ids.append(actual_conv_id)

        assert len(self.conversation) == len(self.conversation_ids)

        cur_utter = []
        for utter in dialogues:
            utter = self.remove_name(utter)
            doc = nlp(utter)

            cur_utter.append(utter)
            code_choice_list = [self.is_code_choice(token, actual_conv_id) for token in doc if token.text != ' ']
            lang = self.get_utterance_type(code_choice_list)

            self.compute_vocabs(doc, actual_conv_id, lang)
            self.total_utter.add_ref_index((len(self.conversation) - 1, len(cur_utter) - 1, len(code_choice_list)),
                                           actual_conv_id)

            if not lang:
                self.other_utter.add_ref_index((len(self.conversation) - 1, len(cur_utter) - 1, len(code_choice_list)),
                                               actual_conv_id)
            elif lang == 'English':
                self.eng_utter.add_ref_index((len(self.conversation) - 1, len(cur_utter) - 1, len(code_choice_list)),
                                             actual_conv_id)

            elif lang == 'Hindi':
                self.native_utter.add_ref_index((len(self.conversation) - 1, len(cur_utter) - 1, len(code_choice_list)),
                                                actual_conv_id)
            elif lang == 'code_mixed':
                self.cm_utter.add_ref_index((len(self.conversation) - 1, len(cur_utter) - 1, len(code_choice_list)),
                                            actual_conv_id)
            else:
                self.other_utter.add_ref_index((len(self.conversation) - 1, len(cur_utter) - 1, len(code_choice_list)),
                                               actual_conv_id)

    def create_cm_sents(self):
        """
        Calculates the total number of code-mixed sentences, Hindi sentences, and English
        sentences in the corpus. This function must be called before calculating any metrics on matrix level
        such as: insertions(), alternations(), etc.
        """

        for i in range(len(self.cm_utter.index_ref_utter)):
            doc = nlp(self.cm_utter.ref_utter[self.cm_utter.index_ref_utter[i][0]][self.cm_utter.index_ref_utter[i][1]])
            for sent in doc.sents:
                code_choice_list = [self.is_code_choice(token, self.cm_utter.actual_ids[i]) for token in sent if
                                    token.text != ' ']
                if self.get_utterance_type(code_choice_list) == 'code_mixed':
                    self.cm_sents.add_sent(self.remove_name(sent.text), self.cm_utter.actual_ids[i])

        self.eng_matrix = Sentence.from_ref_sent(self.cm_sents.sents)
        self.hindi_matrix = Sentence.from_ref_sent(self.cm_sents.sents)
        self.matrix_language_dist()

    def matrix_language_dist(self):
        """
        Calculates the number of sentences with Hindi as the matrix lang and as well as English.
        """
        for i in range(len(self.cm_sents.sents)):
            self.matrix_lang_of_sents(self.cm_sents.sents[i][0], self.cm_sents.actual_ids[i], i)

    def matrix_lang_of_sents(self, sent, actual_conv_id, ref_id_sent, add=True):
        """
        Calculates the matrix language of the sentence.
        Args:
            sent: sentence
            actual_conv_id: conversation id as in the corpus
            ref_id_sent: id of the conversation as stored in the Sentence() object.
        """
        operator_flag = 0
        bigram_flag = 0
        lang_code = []

        doc = nlp(sent)
        sentence_tokens = [token for token in doc if token.text != ' ']

        for i in range(len(sentence_tokens)):
            lang_code.append(self.is_code_choice(sentence_tokens[i], actual_conv_id))

            # Checking if word present in Hindi operator verbs
            if str(sentence_tokens[i].text) in operator_verbs and operator_flag != 1:
                operator_flag = 1

            # Checking if word satisfies bigram condition
            if i + 1 < len(sentence_tokens) and bigram_flag != 1:
                # if str(utterance[i]) in wordbigrams_first:
                for w in range(len(self.wordbigrams) - 1):
                    if self.wordbigrams[w][0] == sentence_tokens[i].text and self.wordbigrams[w][1] == sentence_tokens[
                        i + 1].text:
                        bigram_file = 1
                        break

        # Hindi Majority
        if lang_code.count('Hindi') > len(lang_code) / 2:
            if add:
                self.hindi_matrix.add_ref_index((ref_id_sent, 0, len(sentence_tokens)), actual_conv_id)
            else:
                return 'Hindi'

        # English Majority
        elif lang_code.count('English') > len(lang_code) / 2:
            if operator_flag == 1:
                if add:
                    self.hindi_matrix.add_ref_index((ref_id_sent, 0, len(sentence_tokens)), actual_conv_id)
                else:
                    return 'Hindi'
            elif bigram_flag == 1:
                if add:
                    self.hindi_matrix.add_ref_index((ref_id_sent, 0, len(sentence_tokens)), actual_conv_id)
                else:
                    return 'Hindi'
            else:
                if add:
                    self.eng_matrix.add_ref_index((ref_id_sent, 0, len(sentence_tokens)), actual_conv_id)
                else:
                    return 'English'

        else:
            if operator_flag == 1:
                if add:
                    self.hindi_matrix.add_ref_index((ref_id_sent, 0, len(sentence_tokens)), actual_conv_id)
                else:
                    return 'Hindi'
            elif bigram_flag == 1:
                if add:
                    self.hindi_matrix.add_ref_index((ref_id_sent, 0, len(sentence_tokens)), actual_conv_id)
                else:
                    return 'Hindi'
            else:
                if add:
                    self.eng_matrix.add_ref_index((ref_id_sent, 0, len(sentence_tokens)), actual_conv_id)
                else:
                    return 'English'

    def native_x(self, i, actual_conv_id):
        """
        Calculates the number of Hindi tokens in the given sentence.
        """
        _id = actual_conv_id
        words = nlp(i)
        tln = 0
        for word in words:
            if self.is_code_choice(word, _id) == 'Hindi':
                tln += 1
        if tln > 0:
            return tln
        return self.N_x(i, actual_conv_id)

    def p_x(self, sent, actual_conv_id):
        """
        Calculates the number of switch points in the given sentence.
        """
        _id = actual_conv_id
        words = nlp(sent)
        count = 0
        if len(words) == 1:
            return 0
        prev = ''
        for word in words:
            icc = self.is_code_choice(word, _id)
            if icc in ['English', 'Hindi']:
                if prev == '':
                    prev = icc
                elif prev != icc:
                    count += 1
                    prev = icc
        return count

    def delta_x(self, i, actual_conv_id, prev):
        if prev == '':
            prev = self.matrix_lang_of_sents(i, actual_conv_id, ref_id_sent=-1, add=False)
        elif prev != self.matrix_lang_of_sents(i, actual_conv_id, ref_id_sent=-1, add=False):
            return 1, prev
        return 0, prev

    def N_x(self, sent, actual_conv_id):
        _id = actual_conv_id
        words = nlp(sent)
        counter = 0
        for word in words:
            if self.is_code_choice(word, actual_conv_id) in ['English', 'Hindi']:
                counter += 1
        return counter

    def inner_func(self, i, actual_conv_id, prev):
        d_x, prev = self.delta_x(i, actual_conv_id, prev)
        N = self.N_x(i, actual_conv_id)
        if not N:
            return 0, prev
        eq = 1 + d_x - (self.native_x(i, actual_conv_id) + self.p_x(i, actual_conv_id)) / N
        return eq, prev

    def max_tln(self, sent, actual_conv_id):
        _id = actual_conv_id
        i = sent
        words = nlp(i)
        tln_hindi = 0
        tln_eng = 0
        for word in words:
            lang = self.is_code_choice(word, _id)
            if lang == 'Hindi':
                tln_hindi += 1
            if lang == 'English':
                tln_eng += 1
        return max(tln_eng, tln_hindi)

    def integration_index(self, sent, actual_conv_id):
        _id = actual_conv_id
        i = sent
        words = nlp(i)
        count = 0
        if len(words) == 1:
            return 0
        prev = ''
        for word in words:
            if self.is_code_choice(word, _id) in ['English', 'Hindi']:
                if prev == '':
                    prev = self.is_code_choice(word, _id)
                elif prev != self.is_code_choice(word, _id):
                    count += 1
                    prev = self.is_code_choice(word, _id)
        return count / (len(words) - 1)

    def code_mixed_statistics(self):
        """
        code_mixed_statistics() calculates corpus level statistics such as: CMI-Index (Cc and Cavg), and I-index
        """

        # Collecting all sentences of the corpus
        for i in range(len(self.total_utter.index_ref_utter)):
            doc = nlp(self.total_utter.ref_utter[self.total_utter.index_ref_utter[i][0]][
                          self.total_utter.index_ref_utter[i][1]])
            for sent in doc.sents:
                self.all_sents.add_sent(self.remove_name(sent.text), self.total_utter.actual_ids[i])

        # 1. Cu(x)
        cu_list = []
        for i in range(self.all_sents.count()):
            sent = self.all_sents.sents[i][0]
            if not sent:
                continue
            n_x = self.N_x(sent, self.all_sents.actual_ids[i])
            if n_x:
                result = ((n_x - self.max_tln(sent, self.all_sents.actual_ids[i]) + self.p_x(sent,
                                                                                             self.all_sents.actual_ids[
                                                                                                 i])) / (2 * n_x)) * 100
            else:
                result = 0
            cu_list.append(result)

        ##2. Cavg
        Cu_avg = sum(cu_list) / len(cu_list)
        self.cm_statistics["Cu_avg"] = round(Cu_avg, 2)

        ##3. Cc
        sigma = 0
        prev = ''
        for i in range(self.all_sents.count()):
            sent = self.all_sents.sents[i][0]
            if not sent:
                continue
            result = self.inner_func(sent, self.all_sents.actual_ids[i], prev)
            sigma += result[0]
            prev = result[1]
        sigma *= 0.5
        cu_metric = (100 / self.total_utter.count()) * (sigma + ((5 / 6) * (self.cm_utter.count())))

        self.cm_statistics["cu_metric"] = round(cu_metric, 2)

        # 4. I index
        sum_i_index = 0
        for i in range(len(self.conversation)):
            tmp_sum = 0
            len_conversation = 0
            for utter in self.conversation[i]:
                if utter:
                    len_conversation += 1
                    tmp_sum += self.integration_index(utter, self.conversation_ids[i])
            avg_conversation = tmp_sum / len_conversation
            sum_i_index += avg_conversation
        avg_i_index = sum_i_index / len(self.conversation)
        self.cm_statistics["I_index"] = round(avg_i_index, 2)

    def insertions_and_alternations(self, eng_insertions_in_hindi=False, hin_insertions_in_english=False,
                                    alternations=False, insertions_distributions=False):

        if eng_insertions_in_hindi:
            self.eng_insertions = self.insertions(embedding_lang='English')

        if hin_insertions_in_english:
            self.hindi_insertions = self.insertions(embedding_lang='Hindi')

        if alternations:
            self.eng_alternations = self.alternations(embedding_code='English')
            self.hindi_alternations = self.alternations(embedding_code='Hindi')

        if insertions_distributions:
            self.insertions_distributions(embedding_code="English")
            self.insertions_distributions(embedding_code="Hindi")

    def insertions_distributions(self, embedding_code="English"):
        """
        Calculates the number of single word insertion vs multi-word insertions in code-mixed sentences.
        Args:
            embedding_code: should be either Hindi or English, embedding_code is the language of the
            insertion word/words.
        """

        single_word_insertions = set()
        multi_word_insertions = set()

        if embedding_code == 'English':
            for i in range(self.hindi_matrix.count()):
                sent_text = self.hindi_matrix.ref_sents[self.hindi_matrix.index_ref_sents[i][0]][
                    self.hindi_matrix.index_ref_sents[i][1]]
                sent = nlp(sent_text)
                count, multi_ins = 0, False

                for token in sent:
                    if self.is_code_choice(token, self.hindi_matrix.actual_ids[i]) == embedding_code:
                        count += 1
                    else:
                        count = 0
                    if count == 2:
                        multi_ins = True
                        multi_word_insertions.add(sent_text)
                        break

                if not multi_ins:
                    single_word_insertions.add(sent_text)
            print("Number of sentences with single word insertion(English) : {}".format(len(single_word_insertions)))
            print("Number of sentences with multi word insertion(English) : {}\n".format(len(multi_word_insertions)))

        if embedding_code == 'Hindi':
            for i in range(self.eng_matrix.count()):
                sent_text = self.eng_matrix.ref_sents[self.eng_matrix.index_ref_sents[i][0]][
                    self.eng_matrix.index_ref_sents[i][1]]
                sent = nlp(sent_text)
                count, multi_ins = 0, False

                for token in sent:
                    if self.is_code_choice(token, self.eng_matrix.actual_ids[i]) == embedding_code:
                        count += 1
                    else:
                        count = 0
                    if count == 2:
                        multi_ins = True
                        multi_word_insertions.add(sent_text)
                        break

                if not multi_ins:
                    single_word_insertions.add(sent_text)

            print("Number of sentences with single word insertion(Hindi) : {}".format(len(single_word_insertions)))
            print("Number of sentences with multi word insertion(Hindi) : {}\n".format(len(multi_word_insertions)))

    def alternations(self, embedding_code='English'):
        """
        Calculates the number of alternations(one of the types of code-mixing) in code-mixed sentences
        Args:
            embedding_code: should be either Hindi or English, embedding_code is the language of the
            insertion word/words.
        """

        alternations = set()
        if embedding_code == 'English':
            for i in range(self.hindi_matrix.count()):
                sent_text = self.hindi_matrix.ref_sents[self.hindi_matrix.index_ref_sents[i][0]][
                    self.hindi_matrix.index_ref_sents[i][1]]
                sent = nlp(sent_text)
                count = 0
                for token in sent:
                    if self.is_code_choice(token, self.hindi_matrix.actual_ids[i]) == embedding_code:
                        count += 1
                    else:
                        count = 0  # Reset count
                    if count > 2:
                        alternations.add(sent_text)
                        break

            return alternations

        if embedding_code == 'Hindi':
            for i in range(self.eng_matrix.count()):
                sent_text = self.eng_matrix.ref_sents[self.eng_matrix.index_ref_sents[i][0]][
                    self.eng_matrix.index_ref_sents[i][1]]
                sent = nlp(sent_text)
                count = 0
                for token in sent:
                    if self.is_code_choice(token, self.eng_matrix.actual_ids[i]) == embedding_code:
                        count += 1
                    else:
                        count = 0  # Reset count
                    if count > 2:
                        alternations.add(sent_text)
                        break

            return alternations

    def insertions(self, embedding_lang='English'):
        """
        Calculates the number of code-mixed sentences that have insertions as the type of the code-mixing
        present.
        Args:
            embedding_lang: should be either Hindi or English, embedding_code is the language of the
            insertion word/words.
        """

        insertions = set()
        if embedding_lang == 'English':
            for i in range(self.hindi_matrix.count()):
                alternation = False
                sent_text = self.hindi_matrix.ref_sents[self.hindi_matrix.index_ref_sents[i][0]][
                    self.hindi_matrix.index_ref_sents[i][1]]
                sent = nlp(sent_text)
                count_insertion = 0
                for token in sent:
                    if self.is_code_choice(token, self.hindi_matrix.actual_ids[i]) == embedding_lang:
                        count_insertion += 1
                    else:
                        count_insertion = 0  # Reset count
                    if count_insertion > 2:
                        alternation = True
                        break
                if not alternation:
                    insertions.add(sent_text)

            return insertions

        if embedding_lang == 'Hindi':
            for i in range(self.eng_matrix.count()):
                alternation = False

                sent_text = self.eng_matrix.ref_sents[self.eng_matrix.index_ref_sents[i][0]][
                    self.eng_matrix.index_ref_sents[i][1]]
                sent = nlp(sent_text)
                count_insertion = 0
                for token in sent:
                    if self.is_code_choice(token, self.hindi_matrix.actual_ids[i]) == embedding_lang:
                        count_insertion += 1
                    else:
                        count_insertion = 0  # Reset count
                    if count_insertion > 2:
                        alternation = True
                        break
                if not alternation:
                    insertions.add(sent_text)

            return insertions

    def embedding_words_in_sentence_distribution(self, embedding='Hindi', matrix='English', save_words=True):
        """
        Calculates the distribution of number of words(embedding language) present in code-mixed sentences
        as insertions.
        """
        if embedding == 'Hindi' and matrix == 'English':

            hindi_embedding_in_english_matrix_count = {}
            hindi_embedding_in_english_matrix_sentences = {}

            for i in range(self.eng_matrix.count()):
                hindi_embedding_count = 0
                sent = nlp(self.eng_matrix.ref_sents[self.eng_matrix.index_ref_sents[i][0]][
                               self.eng_matrix.index_ref_sents[i][1]])
                sent_text = self.eng_matrix.ref_sents[self.eng_matrix.index_ref_sents[i][0]][
                    self.eng_matrix.index_ref_sents[i][1]]
                for word in sent:
                    if self.is_code_choice(word, self.eng_matrix.actual_ids[i]) == "Hindi":
                        hindi_embedding_count += 1
                if hindi_embedding_count in hindi_embedding_in_english_matrix_count:
                    hindi_embedding_in_english_matrix_count[hindi_embedding_count] += 1
                else:
                    hindi_embedding_in_english_matrix_count[hindi_embedding_count] = 1

                if save_words:
                    if hindi_embedding_count in hindi_embedding_in_english_matrix_sentences:
                        hindi_embedding_in_english_matrix_sentences[hindi_embedding_count].append(sent_text)
                    else:
                        hindi_embedding_in_english_matrix_sentences[hindi_embedding_count] = [sent_text]

            print("**** Distribution of number of Hindi insertions in English matrix sentences: ****")
            for k, v in hindi_embedding_in_english_matrix_count.items():
                print("Number of sentences with {} insertions: {}".format(k, v))

        if embedding == 'English' and matrix == 'Hindi':

            eng_embedding_in_hindi_matrix_count = {}
            eng_embedding_in_hindi_matrix_sentences = {}

            for i in range(self.hindi_matrix.count()):
                eng_embedding_count = 0
                sent = nlp(self.hindi_matrix.ref_sents[self.hindi_matrix.index_ref_sents[i][0]][
                               self.hindi_matrix.index_ref_sents[i][1]])
                sent_text = self.hindi_matrix.ref_sents[self.hindi_matrix.index_ref_sents[i][0]][
                    self.hindi_matrix.index_ref_sents[i][1]]
                for word in sent:
                    if self.is_code_choice(word, self.hindi_matrix.actual_ids[i]) == "English":
                        eng_embedding_count += 1
                if eng_embedding_count in eng_embedding_in_hindi_matrix_count:
                    eng_embedding_in_hindi_matrix_count[eng_embedding_count] += 1
                else:
                    eng_embedding_in_hindi_matrix_count[eng_embedding_count] = 1

                if save_words:
                    if eng_embedding_count in eng_embedding_in_hindi_matrix_sentences:
                        eng_embedding_in_hindi_matrix_sentences[eng_embedding_count].append(sent_text)
                    else:
                        eng_embedding_in_hindi_matrix_sentences[eng_embedding_count] = [sent_text]

            print("**** Distribution of number of English insertions in Hindi matrix sentences: ****")
            for k, v in eng_embedding_in_hindi_matrix_count.items():
                print("Number of sentences with {} insertions: {}".format(k, v))

    def k_nonnative_words_in_cm_sents(self, k=6, save_words=False, visualization=False):
        """
        Calculates the distribution of the number of english words(1,2,..,k) present in the code-mixed
        sentences that have Hindi as the matrix language.
        Args:
            k: K can be any integer, for our corpus we have kept 6 as the upperbound.
            visualization: If True, saves a bar graph for this distribution.
        """
        k_non_native_dict = {}
        for i in range(k + 1):
            k_non_native_dict[i] = 0

        k_non_native_ls_utterances = {}
        for i in range(k + 1):
            k_non_native_ls_utterances[i] = []

        for i in range(self.cm_sents.count()):
            eng_word_count = 0
            for word in nlp(self.cm_sents.sents[i][0]):
                if self.is_code_choice(word, self.cm_sents.actual_ids[i]) == 'English':
                    eng_word_count += 1
            if 1 <= eng_word_count <= k:
                k_non_native_dict[eng_word_count] += 1
            if save_words:
                k_non_native_ls_utterances[eng_word_count].append(self.cm_sents.sents[i])

        print('\nK non-native(english) words distribution in the code-mixed corpora:')
        print("k, sentences:")
        for i in range(len(k_non_native_dict.items())):
            print(list(k_non_native_dict.items())[i])

        if visualization:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.bar(k_non_native_dict.keys(), k_non_native_dict.values(), 0.8, color="#000000")
            plt.ylabel("Number of utterances")
            plt.xlabel("K value")
            plt.title(f"Plotting {k} non native words in utterances in the corpus")
            plt.show()
            fig.savefig(self.output_dir + "/k_non_native_words.png")
            print("Figure for k non-native words distribution is saved in :{}\n".format(self.output_dir))

    def get_utterance_type(self, code_choice_list):
        """Returns the type of the utterance given the language tags per token,
        the utterance can be from one of the following types [English, Hindi, Code-mixed]
        """

        filtered_list = []  ### example : "thank you ji" --- [english, english, hindi]
        for code in code_choice_list:
            if code in ['English', 'Hindi']:
                filtered_list.append(code)
        if filtered_list:
            utt_type = set(filtered_list)
            if len(utt_type) == 2:
                return 'code_mixed'
            return tuple(utt_type)[
                0]  ### using[0] returns string and not the set - for example : "thank you"---"english[0],english[1]"
        return None


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    # Adding Arguments
    ap.add_argument("-data_dir", "--data_dir", required=True, type=str, help='path to the GupShup dataset')
    ap.add_argument("-hindi_tags", "--hindi_tags", required=True, type=str, help='path to the hindi tags file')
    ap.add_argument("-english_tags", "--english_tags", required=True, type=str, help='path to the english tags file')
    ap.add_argument("-ne_tags", "--ne_tags", required=True, type=str, help='path to the ne tags file')
    ap.add_argument("-hi_Latn_wordbigrams", "--hi_Latn_wordbigrams", required=True, type=str,
                    help='path to the hi_Latn_wordbigrams file')
    ap.add_argument("-output_dir", "--output_dir", required=True, type=str, help='output_dir directory')
    ap.add_argument("-n", "--n", required=True, type=int, help='number of conversations to be used')

    args = vars(ap.parse_args())

    for ii, item in enumerate(args):
        print(item + ': ' + str(args[item]))

    codemixed = CodeMixedStats(args["hindi_tags"], args["english_tags"], args["ne_tags"], args["data_dir"],
                               args["hi_Latn_wordbigrams"], args["output_dir"])

    with open(args["data_dir"], 'r') as fp_json:
        data = json.load(fp_json)
        if args["n"] == -1:
            n = len(data)
        else:
            n = args["n"]
        for i in data[:n]:
            codemixed.get_total_utterances_and_vocab(i['conv'], i['id'])

    print("------------ Dataset statistics ------------")
    print("Total number of conversations: {}".format(len(codemixed.conversation)))
    print("Total number of utterances: {}".format(codemixed.total_utter.count()))
    print("Total number of code-mixed utterances: {}".format(codemixed.cm_utter.count()))
    print("Total number of Hindi utterances: {}".format(codemixed.native_utter.count()))
    print("Total number of English utterances: {}\n".format(codemixed.eng_utter.count()))

    sum_lengths = sum([i[2] for i in codemixed.total_utter.index_ref_utter])
    avg_len_utter = sum_lengths / codemixed.total_utter.count()
    print("Average length of utterances in the corpus: {}".format(int(avg_len_utter)))

    avg_number_utter_per_conversation = codemixed.total_utter.count() / len(codemixed.conversation)
    print("Average number of utterances per conversation: {}".format(int(avg_number_utter_per_conversation)))

    avg_cm_utter_per_conversation = codemixed.cm_utter.count() / len(codemixed.conversation)
    print("Average number of code-mixed utterances per conversation: {}".format(int(avg_cm_utter_per_conversation)))

    percentage_cm_utter = codemixed.cm_utter.count() / codemixed.total_utter.count() * 100
    print("Percentage of code-mixed utterances in the corpus: {}%".format(round(percentage_cm_utter, 2)))

    percentage_native_utter = codemixed.native_utter.count() / codemixed.total_utter.count() * 100
    print("Percentage of Hindi utterances in the corpus: {}%".format(round(percentage_native_utter, 2)))

    percentage_eng_utter = codemixed.eng_utter.count() / codemixed.total_utter.count() * 100
    print("Percentage of English utterances in the corpus: {}%\n".format(round(percentage_eng_utter, 2)))

    print("Total vocabulary size: {}".format(codemixed.total_vocab.count()))
    print("Total English vocabulary size: {}".format(codemixed.eng_vocab.count()))
    print("Total code-mixed English vocabulary size: {}".format(codemixed.cm_eng_vocab.count()))
    print("Total Hindi vocabulary size: {}".format(codemixed.native_vocab.count()))
    print("Total ner vocabulary size: {}".format(codemixed.ner_vocab.count()))
    print("Total other vocabulary size: {}\n".format(codemixed.other_vocab.count()))

    codemixed.create_cm_sents()  # Needed for all the matrix level metrics

    print("Total number of code-mixed sentences: {}".format(codemixed.cm_sents.count()))
    print("Total number of sentences with Hindi matrix: {}".format(codemixed.hindi_matrix.count()))
    print("Total number of sentences with English matrix: {}\n".format(codemixed.eng_matrix.count()))

    codemixed.insertions_and_alternations(eng_insertions_in_hindi=True, hin_insertions_in_english=False,
                                          alternations=False, insertions_distributions=False)
    print("Number of sentences with English insertions: {}".format(len(codemixed.eng_insertions)))

    codemixed.insertions_and_alternations(eng_insertions_in_hindi=False, hin_insertions_in_english=True,
                                          alternations=False, insertions_distributions=False)
    print("Number of sentences with Hindi insertions: {}\n".format(len(codemixed.hindi_insertions)))

    codemixed.insertions_and_alternations(eng_insertions_in_hindi=False, hin_insertions_in_english=False,
                                          alternations=True, insertions_distributions=False)
    print("Number of sentences with English alternations: {}".format(len(codemixed.eng_alternations)))
    print("Number of sentences with Hindi alternations: {}\n".format(len(codemixed.hindi_alternations)))

    codemixed.insertions_and_alternations(eng_insertions_in_hindi=False, hin_insertions_in_english=False,
                                          alternations=False, insertions_distributions=True)

    codemixed.embedding_words_in_sentence_distribution(embedding='Hindi', matrix='English', save_words=True)
    codemixed.embedding_words_in_sentence_distribution(embedding='English', matrix='Hindi', save_words=True)

    codemixed.k_nonnative_words_in_cm_sents(k=6, save_words=False, visualization=True)

    codemixed.code_mixed_statistics()
    print("------------ Corpus level statistics ------------")
    for k, v in codemixed.cm_statistics.items():
        print(k + " : " + str(v))
