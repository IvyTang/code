import spacy
from collections import namedtuple

Token = namedtuple('Token', ['text', 'lemma_', 'pos_', 'tag_'])


class Lemmatizer(object):
    def __init__(self):
        self._lemmatizer = spacy.load('en')

    def post_process_tokens(self, tokens):
        tokens_processed = []
        for token in tokens:
            token_my = Token(text=token.text, lemma_=token.lemma_, pos_=token.pos_, tag_=token.tag_)
            tokens_processed.append(token_my)
        i = 1
        while i < len(tokens_processed) - 1:
            token = tokens_processed[i]
            token_pred = tokens_processed[i - 1]
            token_succ = tokens_processed[i + 1]
            if (token.text == 'num' or token.text == 'pun') and token_pred.text == '<' and token_succ.text == '>':
                del tokens_processed[i + 1]
                text = '<' + token.text + '>'
                token_my = Token(text=text, lemma_=text, pos_=text, tag_=text)
                tokens_processed[i] = token_my
                del tokens_processed[i - 1]
                i -= 1
            i += 1
        return tokens_processed
        
    def post_process_word(self, original_word, lemmatized_word):
        if lemmatized_word == '-PRON-':
            return original_word
        if original_word.isupper():
            return lemmatized_word.upper()
        if original_word.istitle():
            return lemmatized_word.capitalize()
        return lemmatized_word

    def get_lemmatized_idx(self, tokens):
        lemmatized_idx_array = []
        for i in range(len(tokens)):
            token = tokens[i]
            if token.pos_ in {'VERB', 'ADJ', 'ADV', 'NOUN'}:
                lemmatized_idx_array.append(int(i))
            # else:
            #     if token.text.lower() != token.lemma_.lower() and token.lemma_ != '-PRON-':
            #         print(token.text, token.lemma_, token.pos_, token.tag_)
        return lemmatized_idx_array

    def lemmatize(self, sentence):
        sentence = sentence.strip()
        tokens = self._lemmatizer(sentence)
        tokens_processed = self.post_process_tokens(tokens)
        words_original = [token.text for token in tokens_processed]
        words_pos = [token.pos_ for token in tokens_processed]
        words_lemmatized = [self.post_process_word(token.text, token.lemma_) for token in tokens_processed]
        lemmatized_idx_array = self.get_lemmatized_idx(tokens_processed)
        return words_original, words_lemmatized, words_pos, lemmatized_idx_array


if __name__ == "__main__":
    lemmatizer = Lemmatizer()
    # sentence = '" Most of the above arguments may be found much more at large in Burke \'s \' Vindication of Natural Society \' ; a treatise in which the evils of the existing political institutions are displayed with incomparable force of reasoning and lustre of eloquence . " – footnote , Ch .'
    sentence = "boeing"
    # sentence = '<pun> this is a <num>'
    # sentence = 'he'
    # sentence = 'Autism affects information processing in the brain by altering how nerve cells and their synapses connect and organize ; how this occurs is not well understood'
    # sentence = 'The Blackwell Dictionary of Modern Social Thought'
    # sentence = 'We\'re here'
    # sentence = 'I\'m here for your life.'
    sentence_original, sentence_lemmatized, words_pos, lemmatized_idx = lemmatizer.lemmatize(sentence)
    print(sentence)
    print(sentence_original)
    print(sentence_lemmatized)
    print(words_pos)
    print(lemmatized_idx)
