import csv
import regex as re
import json
import pickle
import sys

choose_flag = '.+choose\d:.+'
slide_flag = '.+slide\d*:.+'
input_re = 'input:.+'
choose_re = 'choose\d:.+'
# punctuation_re = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
punctuation_re = '[’!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~]'
word_re = '[a-zA-Z]'
debug_log = False
print_log = False
save_word_trace_all = True


class InputLetter:
    x = None
    y = None
    input_letter = None

    def __init__(self, x, y, input_letter):
        self.x = x
        self.y = y
        self.input_letter = input_letter

    def __str__(self):
        return str(self.x) + ',' + str(self.y) + ',' + str(self.input_letter)


# 每个单词trace数据
class WordTrace:
    # 正常输入的数据
    input_word = None
    input_letter_list = None
    suggest_words = None
    choose_word = None
    choose_word_idx = None
    choose_type = None
    choose_language_detection = None

    # 回删后重新输入的数据
    change_word = None
    change_letter_list = None
    change_suggest_words = None
    change_choose_word = None
    change_choose_word_idx = None
    change_choose_type = None
    change_choose_language_detection = None

    is_finish = False  # 标记单词是否输入完成
    is_choose = False

    def __init__(self):
        self.input_letter_list = []
        self.change_letter_list = []

    def __str__(self):
        return str(self.input_word) + '|' + str(self.suggest_words) + '|' + str(self.choose_word) + '|' + \
               str(self.choose_type) + '|' + str(self.change_word) + '|' + str(self.change_suggest_words) + \
               '|' + str(self.change_choose_word) + '|' + str(self.change_choose_type)

    def print_info(self):
        print("\tinput_word: " + str(self.input_word) +
              "\tsuggest_words: " + str(self.suggest_words) +
              "\tchoose_word: " + str(self.choose_word) +
              "\tchoose_word_idx: " + str(self.choose_word_idx) +
              "\tchoose_type: " + str(self.choose_type) +
              "\tchoose_language_detection: " + str(self.choose_language_detection) +
              "\n" +
              "\tinput_letter_list: " + " ".join([str(c) for c in self.input_letter_list]) +
              "\n" +
              "\tchange_word: " + str(self.change_word) +
              "\tchange_suggest_words: " + str(self.change_suggest_words) +
              "\tchange_choose_word: " + str(self.change_choose_word) +
              "\tchange_choose_word_idx: " + str(self.change_choose_word_idx) +
              "\tchange_choose_type: " + str(self.change_choose_type) +
              "\tchange_choose_language_detection: " + str(self.change_choose_language_detection) +
              "\n" +
              "\tchange_letter_list: " + " ".join([str(c) for c in self.change_letter_list]) +
              "\n" +
              "\tis_finish: " + str(self.is_finish) +
              "\tis_choose: " + str(self.is_choose) +
              "\n")

    __repr__ = __str__


class WordTraceArray:
    word_trace_list = None
    kh = None
    kl = None
    kw = None
    rnn_model_version = None

    def __init__(self, kh, kl, kw, rnn_model_version):
        self.word_trace_list = []
        self.kh = kh
        self.kl = kl
        self.kw = kw
        self.rnn_model_version = rnn_model_version

    def print_info(self):
        print ("\tkh: %s\tkl: %s\tkw: %s\trnn_model_version: %s"%(self.kh, self.kl, self.kw, self.rnn_model_version))
        for word_trace in self.word_trace_list:
            word_trace.print_info()


def get_extra_information(extra):
    extra_json = json.loads(extra)
    kh, kl, kw, rnn_model_version = None, None, None, None
    if "extra" in extra_json:
        extra_json = extra_json["extra"]
        if "kh" in extra_json:
            kh = extra_json["kh"]
        if "kl" in extra_json:
            kl = extra_json["kl"]
        if "kw" in extra_json:
            kw = extra_json["kw"]
        if "rnn_model_version" in extra_json:
            rnn_model_version = extra_json["rnn_model_version"]
    return kh, kl, kw, rnn_model_version


def append_word_trace(word_trace_list, word_trace):
    split_word, last_letter_list, word = [], [], ''
    if word_trace.change_word:
        word = re.sub('(' + punctuation_re + r'+)', r" \1 ", word_trace.change_word).strip()
        last_letter_list = word_trace.change_letter_list
    elif word_trace.input_word:
        word = re.sub('(' + punctuation_re + r'+)', r" \1 ", word_trace.input_word).strip()
        last_letter_list = word_trace.input_letter_list

    split_word = re.split(' +', word)
    if len(split_word) != 2:
        word_trace_list.append(word_trace)
        return

    if word_trace.change_word:
        word_trace.change_word = split_word[0]
        word_trace.change_letter_list = last_letter_list[:-len(split_word[1])]
    elif word_trace.input_word:
        word_trace.input_word = split_word[0]
        word_trace.input_letter_list = last_letter_list[:-len(split_word[1])]
    word_trace_list.append(word_trace)
    word_trace = WordTrace()
    word_trace.choose_word = word_trace.input_word = split_word[1]
    word_trace.input_letter_list = last_letter_list[-len(split_word[1]):]
    word_trace_list.append(word_trace)


def parse(input_file, emoji_re, file_type="csv"):
    word_trace_array_all = []
    word_trace_array_match = []
    num_no_choose = 0
    # count = 0
    with open(input_file, 'r', encoding='UTF-8') as f:
        if file_type == "csv":
            lines = csv.DictReader(x.replace('\0', '') for x in f)
        else:
            lines = f.readlines()
        for line in lines:
            # count += 1
            # if count >= 5000:break
            # print(count)
            extra = '{}'
            if file_type == "csv":
                trace = line['trace']
                word = line['word']
                if 'extra' in line:
                    extra = line['extra']
            else:
                line_split = line.split("\t")
                trace = line_split[4]
                word = line_split[1]
                if line_split[5] != '':
                    extra = line_split[5]

            # 去除不调用引擎的数据和滑行输入数据
            if not re.match(choose_flag, trace) or re.match(slide_flag, trace):
                num_no_choose += 1
                continue

            # num_choose += 1
            trace_array = trace.split(';')
            word_array = word.split()
            word = re.sub('(' + emoji_re + '|' + punctuation_re+ r'+)', r" \1 ", word).strip()
            word_array = re.split(r' +', word)

            if print_log:
              print(word_array)
            kh, kl, kw, rnn_model_version = get_extra_information(extra)
            word_trace_array = WordTraceArray(kh, kl, kw, rnn_model_version)
            word_trace = WordTrace()
            for i in range(1, len(trace_array)):
                trace_one = trace_array[i]
                if len(trace_one) == 0:
                    continue

                # 单个键码为input
                if re.match(input_re, trace_one):
                    input_letter = ''
                    x = None
                    y = None
                    data_array = trace_one.split(',')
                    # ","为trace的分隔符，用户键码不为","的情况
                    if len(data_array) == 4:
                        input_letter = data_array[3]
                        x = data_array[1]
                        y = data_array[2]
                    # ","为trace的分隔符，用户输入建码为","的情况
                    elif len(data_array) == 5:
                        if trace_one.endswith(',,'):
                            input_letter = ','
                            x = data_array[1]
                            y = data_array[2]
                    # 其他情况则认为解析异常
                    else:
                        print('split input trace error:', trace_one)

                    input_letter_obj = InputLetter(x, y, input_letter)
                    # 用户输入建码为键盘上一个字母或符号
                    if len(input_letter) == 1:
                        # 输入建码为空格
                        if input_letter == ' ':
                            # 空格为单词输入完成的标志之一，出现空格时，将is_finish设置为true
                            if not word_trace.is_finish:
                                if word_trace.change_word:
                                    word_trace.change_choose_word = word_trace.change_word
                                else:
                                    word_trace.choose_word = word_trace.input_word
                                word_trace.is_finish = True
                                word_trace.is_choose = False
                            else:
                                word_trace.is_choose = False
                        # 输入建码非空格
                        else:
                            # 当前单词已经输入完成，根据输入建码判断是否保存trace数据并new一个新的对象解析下一个单词
                            if word_trace.is_finish:
                                # 当前建码是标点，则继续添加标点
                                if re.match(punctuation_re, input_letter) and word_trace.is_choose:
                                    if word_trace.change_word:
                                        word_trace.change_word += input_letter
                                        word_trace.change_letter_list.append(input_letter_obj)
                                    else:
                                        word_trace.input_letter_list.append(input_letter_obj)
                                        if word_trace.input_word:
                                            word_trace.input_word += input_letter
                                        else:
                                            word_trace.input_word = input_letter

                                    # word_trace.is_finish = False
                                # 当前建码非标点，则认为开始输入下一个单词，保存trace数据并new一个新的对象解析下一个单词
                                else:
                                    append_word_trace(word_trace_array.word_trace_list, word_trace)
                                    # word_trace_array.word_trace_list.append(word_trace)
                                    word_trace = WordTrace()
                                    word_trace.input_word = input_letter
                                    word_trace.input_letter_list.append(input_letter_obj)
                            # 当前单词没有输入完成，则继续添加键码
                            else:
                                if word_trace.change_word:
                                    word_trace.change_word += input_letter
                                    word_trace.change_letter_list.append(input_letter_obj)
                                else:
                                    word_trace.input_letter_list.append(input_letter_obj)
                                    if word_trace.input_word:
                                        word_trace.input_word += input_letter
                                    else:
                                        word_trace.input_word = input_letter

                    # 用户输入的建码为delete、shift、emoji、capslock等情况
                    else:
                        # 用户输入为delete
                        if input_letter == 'delete':
                            # 单词输入完成，则开始进行回删
                            if word_trace.is_finish:
                                # 已经存在回删，则继续回删
                                if word_trace.change_word:
                                    word_trace.input_word = word_trace.change_word
                                    word_trace.input_letter_list = word_trace.change_letter_list.copy()
                                    word_trace.choose_word = word_trace.change_choose_word
                                    word_trace.choose_type = word_trace.change_choose_type
                                    word_trace.suggest_words = word_trace.change_suggest_words
                                    word_trace.is_finish = False
                                # 不存在回删，则将is_finish设置为false，并设置change_word，开始回删
                                else:
                                    word_trace.is_finish = False
                                    if not word_trace.input_word:
                                        break
                                    word_trace.change_word = word_trace.input_word
                                    word_trace.change_letter_list = word_trace.input_letter_list.copy()
                            # trace没有输入完成，则继续回删
                            else:
                                # 存在change_word则回删change_word
                                if word_trace.change_word:
                                    word_trace.change_word = word_trace.change_word[0:len(word_trace.change_word) - 1]
                                    word_trace.change_letter_list.pop()
                                # 不存在change_word则回删input_word
                                else:
                                    if word_trace.input_word:
                                        word_trace.input_word = word_trace.input_word[0: len(word_trace.input_word) - 1]
                                        if not word_trace.input_letter_list:
                                            print(str(word_trace.input_word))
                                        word_trace.input_letter_list.pop()
                                    else:
                                        if debug_log:
                                            print('delete error:', trace_one, trace)
                        # 用户输入为其他情况
                        else:
                            if not (input_letter == 'symbol' or input_letter == 'shift' or input_letter == 'emoji' or input_letter == 'capslock'):
                                if not (len(input_letter) == 0 or input_letter.startswith(
                                        '-') or input_letter == 'shortcut' or input_letter == 'actionPrevious'):
                                    if debug_log:
                                        print('input_not_delete:', trace_one)

                # 单个键码为choose
                elif re.match(choose_re, trace_one):
                    # 处理连续choose的数据，将上一个单词的trace保存进结果数组，并new一个新的对象解析下一个单词
                    if word_trace.is_finish and word_trace.is_choose:
                        append_word_trace(word_trace_array.word_trace_list, word_trace)
                        # word_trace_array.word_trace_list.append(word_trace)
                        word_trace = WordTrace()
                    data_array = trace_one.split(',')

                    # choose数据解析正常，包括0个及以上的候选词
                    if len(data_array) >= 3:
                        language_detection = data_array[2].split('-')[-1]
                        if language_detection == 'true':
                            language_detection = True
                        elif language_detection != 'false':
                            language_detection = False
                        else:
                            language_detection = None
                        # 有回删的情况下，推荐和上屏结果保存到带有change前缀的成员变量中
                        if word_trace.change_word:
                            word_trace.change_choose_language_detection = language_detection
                            word_trace.change_suggest_words = data_array[3:]
                            word_trace.change_choose_word_idx = int(data_array[1])
                            # 分别处理idx是否为-1的情况，-1表示直接上屏输入的词
                            if word_trace.change_choose_word_idx == -1:
                                word_trace.change_choose_word = word_trace.change_word
                            elif word_trace.change_choose_word_idx < len(word_trace.change_suggest_words):
                                word_trace.change_choose_word = word_trace.change_suggest_words[word_trace.change_choose_word_idx]
                            word_trace.change_choose_type = int(data_array[0][6])

                            # choose6处理 (标点，数字，emoji上屏)， 需要保留最后的标点/emoji等
                            if word_trace.change_choose_type == 6 and \
                               word_trace.change_choose_word_idx != -1 and \
                               word_trace.choose_word:
                                emoji = re.findall(emoji_re, word_trace.change_word)
                                if emoji:
                                    last_input_letter = word_trace.change_letter_list[-1]
                                    word_trace.change_letter_list.pop()
                                    word_trace.change_word = word_trace.change_word[:-len(emoji[-1])]
                                    append_word_trace(word_trace_array.word_trace_list, word_trace)
                                    # word_trace_array.word_trace_list.append(word_trace)
                                    word_trace = WordTrace()
                                    word_trace.choose_word = word_trace.input_word = emoji[-1]
                                    word_trace.input_letter_list.append(last_input_letter)

                        # 没有回删情况下，推荐和上屏结果保存到没有change前缀的成员变量中
                        else:
                            word_trace.choose_language_detection = language_detection
                            word_trace.suggest_words = data_array[3:]
                            word_trace.choose_word_idx = int(data_array[1])
                            # 分别处理idx是否为-1的情况，-1表示直接上屏输入的词
                            if word_trace.choose_word_idx == -1:
                                word_trace.choose_word = word_trace.input_word
                            elif word_trace.choose_word_idx < len(word_trace.suggest_words):
                                word_trace.choose_word = word_trace.suggest_words[word_trace.choose_word_idx]
                            word_trace.choose_type = int(data_array[0][6])

                            # choose6处理 (标点，数字，emoji上屏)， 需要保留最后的标点/emoji等
                            if word_trace.choose_type == 6 and word_trace.choose_word_idx != -1 and word_trace.input_word:
                                emoji = re.findall(emoji_re, word_trace.input_word)
                                if emoji:
                                    last_input_letter = word_trace.input_letter_list[-1]
                                    word_trace.input_letter_list.pop()
                                    word_trace.input_word = word_trace.input_word[:-len(emoji[-1])]
                                    append_word_trace(word_trace_array.word_trace_list, word_trace)
                                    # word_trace_array.word_trace_list.append(word_trace)
                                    word_trace = WordTrace()
                                    word_trace.choose_word = word_trace.input_word = emoji[-1]
                                    word_trace.input_letter_list.append(last_input_letter)

                    # choose数据解析异常
                    else:
                        word_trace.choose_word_idx = -1
                        word_trace.choose_word = ''
                        word_trace.choose_type = data_array[0][6]
                        if debug_log:
                            print('split choose trace error:', trace_one)
                    word_trace.is_finish = True
                    word_trace.is_choose = True

                # 单个键码trace为其他情况
                else:
                    if debug_log:
                        print('not input and choose trace:', trace_one)

            # 处理完一条trace后，将最后可能未保存的一个单词的trace存入数组
            if word_trace.is_finish:
                append_word_trace(word_trace_array.word_trace_list, word_trace)
                # word_trace_array.word_trace_list.append(word_trace)
            else:
                if word_trace.change_word:
                    word_trace.change_choose_word = word_trace.change_word
                else:
                    word_trace.choose_word = word_trace.input_word
                append_word_trace(word_trace_array.word_trace_list, word_trace)
                # word_trace_array.word_trace_list.append(word_trace)

            if print_log:
                print("Line: " + str(line))

            # 验证解析出的trace和word是否前缀/后缀一致
            all_prefix_match = True
            all_suffix_match = True
            word_trace_list = word_trace_array.word_trace_list
            min_len = min(len(word_trace_list), len(word_array))
            for i in range(min_len):
                prefix_choose_word = word_trace_list[i].change_choose_word if word_trace_list[i].change_choose_word \
                    else word_trace_list[i].choose_word
                if word_array[i] != prefix_choose_word:
                    all_prefix_match = False

                suffix_trace = word_trace_list[len(word_trace_list) - i - 1]
                suffix_word = word_array[len(word_array) - i - 1]
                suffix_choose_word = suffix_trace.change_choose_word if suffix_trace.change_choose_word \
                    else suffix_trace.choose_word
                if suffix_word != suffix_choose_word:
                    all_suffix_match = False

                if not all_prefix_match and not all_suffix_match:
                    break
            if all_prefix_match:
                word_trace_array.word_trace_list = word_trace_list[:min_len]
                word_trace_array_match.append(word_trace_array)
            elif all_suffix_match:
                word_trace_array.word_trace_list = word_trace_list[-min_len:]
                word_trace_array_match.append(word_trace_array)

            if save_word_trace_all:
                word_trace_array_all.append(word_trace_array)

            if print_log:
                print("Matched: " + str(all_prefix_match or all_suffix_match) +
                      "\tAll prefix match: " + str(all_prefix_match) + "\tAll suffix match: " + str(all_suffix_match))

                print('Word trace array:')
                word_trace_array.print_info()
                # input()

    if print_log:
        print("Count of word_trace_array_match/word_trace_array_all: %d/%d"
              %(len(word_trace_array_match), len(word_trace_array_all)))
    return word_trace_array_all, word_trace_array_match


def save(trace_array, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(len(trace_array), f)
        for word_trace in trace_array:
            pickle.dump(word_trace, f)


def check_emoji_not_match(input_words_list, choose_words_list, emoji_re, wrong_keys_set):
    for input_word, choose_word in zip(input_words_list, choose_words_list):
        if not input_word or not choose_word:
            return True
        elif len(input_word) - len(choose_word) >= 3 and input_word.lower() not in wrong_keys_set:
            if choose_word.lower() in input_word.lower():
                print(input_word, choose_word)
                return True

        if input_word != choose_word:
            if re.search(emoji_re, input_word) or re.search(emoji_re, choose_word):
                return True

    return False


def load_vocab(vocab_file, split_flag):
    vocab_dict = dict()
    with open(vocab_file, "r") as f:
        for line in f:
            line = line.strip()
            line_split = re.split(split_flag, line)
            if len(line_split) == 2:
                token, id = line_split
                vocab_dict[token] = int(id)
            elif len(line_split) == 1:
                token = line_split[0]
                vocab_dict[token] = 1
            else:
                print("vocab split error : " + line)
    f.close()
    print("vocab num = " + str(len(vocab_dict)))
    print(vocab_dict)
    return vocab_dict


def load_set(vocab_file, split_flag):
    vocab_set = set()
    with open(vocab_file, "r") as f:
        for line in f:
            line = line.strip()
            line_split = re.split(split_flag, line)

            vocab_set.add(line_split[0])

    f.close()
    print("vocab num = " + str(len(vocab_set)))
    print(vocab_set)
    return vocab_set


def test_file_gen(pickled_file, output_file, emoji_re, wrong_keys_set):
    with open(pickled_file, 'rb') as fi, open(output_file, "w", encoding='UTF-8') as fo:
        word_trace_num = pickle.load(fi)
        count = 0
        for i in range(word_trace_num):
            # print("trace " + str(i))
            word_trace_array = pickle.load(fi)
            input_word_list, choose_word_list = [], []

            for word_trace in word_trace_array.word_trace_list:

                input_word_list.append(word_trace.input_word)
                if word_trace.change_choose_word is not None:
                    choose_word_list.append(word_trace.change_choose_word)
                else:
                    choose_word_list.append(word_trace.choose_word)
            if not check_emoji_not_match(input_word_list, choose_word_list, emoji_re, wrong_keys_set):
            # if None not in input_word_list and None not in choose_word_list:

                fo.write("\t".join([" ".join(input_word) for input_word in input_word_list]) + "|#|" + "\t".join(choose_word_list) + "\n")
    fo.close()
    print("wrong key count:", count)


def gen_emoji_re(emoji_file):
    with open(emoji_file, encoding='UTF-8') as f:
        emoji_regex = f.read().strip().replace('\n', '|')
        return emoji_regex


if __name__ == "__main__":
    args = sys.argv

    # trace_file = args[1]
    trace_file = '../trace_raw/word_trace_en_US_20181208.csv'
    emoji_file = '../vocab/emoji'  # 当前因为正则表达式解析的问题，去掉了带*的emoji
    output_file_all = '../trace_processed/' + trace_file.split("/")[-1] + '.all'
    output_file_match = '../trace_processed/' + trace_file.split("/")[-1] + '.match'
    output_test_file = '../trace_data/' + trace_file.split("/")[-1] + '.data'
    wrong_keys_set = load_set("../vocab/wordMap_en_US_one_year", "\t+")
    emoji_regex = gen_emoji_re(emoji_file)
    word_trace_array_all, word_trace_array_match = parse(trace_file, emoji_regex, file_type='csv')
    save(word_trace_array_all, output_file_all)
    save(word_trace_array_match, output_file_match)
    test_file_gen(output_file_match, output_test_file, emoji_regex, wrong_keys_set)
