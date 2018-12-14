
class ResultCounter:

    def __init__(self, data_util):
        self.data_util = data_util

        self.line_count = 0
        self.code_total_count = 0
        self.emoji_total_count = 0
        self.emoji_emoji_count = 0
        self.word_emoji_count = 0
        self.unk_word_count = 0
        self.und_word_count = 0
        self.unk_emoji_count = 0
        self.word_count = 0

        self.word_top_1_count = 0
        self.word_top_3_count = 0

        self.unk_top_1_match = 0
        self.unk_top_3_match = 0
        self.unk_top_1_total = 0
        self.unk_top_3_total = 0

        self.und_top_1_match = 0
        self.und_top_3_match = 0
        self.und_top_1_total = 0
        self.und_top_3_total = 0

        self.emoji_top_1_count = 0
        self.emoji_top_3_count = 0
        self.emoji_tag_top_1_match = 0
        self.emoji_tag_top_3_match = 0
        self.emoji_tag_top_1_total = 0
        self.emoji_tag_top_3_total = 0

        self.emoji_emoji_top_1_match = 0
        self.emoji_emoji_top_3_match = 0
        self.emoji_emoji_tag_top_1_match = 0
        self.emoji_emoji_tag_top_3_match = 0
        self.emoji_emoji_tag_top_1_total = 0
        self.emoji_emoji_tag_top_3_total = 0

        self.word_emoji_top_1_match = 0
        self.word_emoji_top_3_match = 0
        self.word_emoji_tag_top_1_match = 0
        self.word_emoji_tag_top_3_match = 0
        self.word_emoji_tag_top_1_total = 0
        self.word_emoji_tag_top_3_total = 0

        self.emoji_combi_top_1_total = 0
        self.emoji_combi_top_3_total = 0
        self.emoji_combi_top_1_match = 0
        self.emoji_combi_top_3_match = 0

        self.word_emoji_combi_top_1_total = 0
        self.word_emoji_combi_top_3_total = 0
        self.word_emoji_combi_top_1_match = 0
        self.word_emoji_combi_top_3_match = 0

        self.emoji_emoji_combi_top_1_total = 0
        self.emoji_emoji_combi_top_3_total = 0
        self.emoji_emoji_combi_top_1_match = 0
        self.emoji_emoji_combi_top_3_match = 0

        self.merged_word_top_1_count = 0
        self.merged_word_top_3_count = 0

        self.merged_unk_top_1_match = 0
        self.merged_unk_top_3_match = 0
        self.merged_unk_top_1_total = 0
        self.merged_unk_top_3_total = 0

        self.merged_und_top_1_match = 0
        self.merged_und_top_3_match = 0
        self.merged_und_top_1_total = 0
        self.merged_und_top_3_total = 0

        self.merged_emoji_top_1_count = 0
        self.merged_emoji_top_3_count = 0
        self.merged_emoji_tag_top_1_match = 0
        self.merged_emoji_tag_top_3_match = 0
        self.merged_emoji_tag_top_1_total = 0
        self.merged_emoji_tag_top_3_total = 0

        # self.merged_emoji_combi_top_1_count = 0
        # self.merged_emoji_combi_top_3_count = 0
        # self.merged_emoji_tag_combi_top_1_match = 0
        # self.merged_emoji_tag_combi_top_3_match = 0
        self.merged_emoji_tag_combi_top_1_total = 0
        self.merged_emoji_tag_combi_top_3_total = 0

        self.merged_emoji_emoji_top_1_match = 0
        self.merged_emoji_emoji_top_3_match = 0
        self.merged_emoji_emoji_tag_top_1_match = 0
        self.merged_emoji_emoji_tag_top_3_match = 0
        self.merged_emoji_emoji_tag_top_1_total = 0
        self.merged_emoji_emoji_tag_top_3_total = 0

        self.merged_word_emoji_top_1_match = 0
        self.merged_word_emoji_top_3_match = 0
        self.merged_word_emoji_tag_top_1_match = 0
        self.merged_word_emoji_tag_top_3_match = 0
        self.merged_word_emoji_tag_top_1_total = 0
        self.merged_word_emoji_tag_top_3_total = 0

        self.merged_emoji_combi_top_1_count = 0
        self.merged_emoji_combi_top_3_count = 0
        self.merged_emoji_emoji_combi_top_1_match = 0
        self.merged_emoji_emoji_combi_top_3_match = 0
        self.merged_word_emoji_combi_top_1_match = 0
        self.merged_word_emoji_combi_top_3_match = 0

    def count_word_result(self, target_word, word_result, is_prev_emoji, is_target_emoji):
        if self.data_util.emoji_str == word_result[0]:  # emoji tag top1/top3
            self.emoji_tag_top_1_total += 1
            if is_prev_emoji:
                self.emoji_emoji_tag_top_1_total += 1
            else:
                self.word_emoji_tag_top_1_total += 1
        if self.data_util.emoji_str in word_result:
            self.emoji_tag_top_3_total += 1
            if is_prev_emoji:
                self.emoji_emoji_tag_top_3_total += 1
            else:
                self.word_emoji_tag_top_3_total += 1

        if self.data_util.unk_str == word_result[0]:    # unk top1/top3
            self.unk_top_1_total += 1
        if self.data_util.unk_str in word_result:
            self.unk_top_3_total += 1

        if self.data_util.unk_str2 == word_result[0]:   # und top1/top3
            self.und_top_1_total += 1
        if self.data_util.unk_str2 in word_result:
            self.und_top_3_total += 1

        if is_target_emoji: 
            # top 1 emoji tag
            if self.data_util.emoji_str == word_result[0]:
                self.emoji_tag_top_1_match += 1
                if is_prev_emoji:
                    self.emoji_emoji_tag_top_1_match += 1
                else:
                    self.word_emoji_tag_top_1_match += 1
            # top 3 emoji tag
            if self.data_util.emoji_str in word_result:
                self.emoji_tag_top_3_match += 1
                if is_prev_emoji:
                    self.emoji_emoji_tag_top_3_match += 1
                else:
                    self.word_emoji_tag_top_3_match += 1   
        else:
            if target_word == self.data_util.unk_str:
                # match unknown
                if target_word == word_result[0]:
                    self.unk_top_1_match += 1
                if target_word in word_result:
                    self.unk_top_3_match += 1
            elif target_word == self.data_util.unk_str2:
                # match und
                if target_word == word_result[0]:
                    self.und_top_1_match += 1
                if target_word in word_result:
                    self.und_top_3_match += 1
            else:
                # match word
                if target_word == word_result[0]:
                    self.word_top_1_count += 1
                if target_word in word_result:
                    self.word_top_3_count += 1

    def count_emoji_result(self, target_emoji, emoji_result, is_prev_emoji, is_target_emoji):
        if is_target_emoji:
            # emoji label match
            if target_emoji != self.data_util.unk_emoji:
                if target_emoji == emoji_result[0]:
                    self.emoji_top_1_count += 1
                    if is_prev_emoji:
                        self.emoji_emoji_top_1_match += 1
                    else:
                        self.word_emoji_top_1_match += 1
                if target_emoji in emoji_result:
                    self.emoji_top_3_count += 1
                    if is_prev_emoji:
                        self.emoji_emoji_top_3_match += 1
                    else:
                        self.word_emoji_top_3_match += 1
                
                is_combi = [len(res) > 1 for res in emoji_result]
                if is_combi[0]:
                    self.emoji_combi_top_1_total += 1
                    if is_prev_emoji:
                        self.emoji_emoji_combi_top_1_total += 1
                    else:
                        self.word_emoji_combi_top_1_total += 1
                if True in is_combi:
                    self.emoji_combi_top_3_total += 1
                    if is_prev_emoji:
                        self.emoji_emoji_combi_top_3_total += 1
                    else:
                        self.word_emoji_combi_top_3_total += 1
                
                contains_target = [(target_emoji in res or res in target_emoji) for res in emoji_result]
                if contains_target[0]:
                    self.emoji_combi_top_1_match += 1
                    if is_prev_emoji:
                        self.emoji_emoji_combi_top_1_match += 1
                    else:
                        self.word_emoji_combi_top_1_match += 1
                if True in contains_target:
                    self.emoji_combi_top_3_match += 1
                    if is_prev_emoji:
                        self.emoji_emoji_combi_top_3_match += 1
                    else:
                        self.word_emoji_combi_top_3_match += 1                        
            else:
                self.unk_emoji_count += 1  

    def count_merged_result(self, target_word, target_emoji, merged_result, is_prev_emoji, is_target_emoji):
        # result merged from word and emoji
        is_res_emoji = [self.data_util.is_emoji(r) for r in merged_result]
        is_res_emoji_combi = [False]

        if is_res_emoji[0] or is_res_emoji_combi[0]:
            self.merged_emoji_tag_top_1_total += 1
            if is_prev_emoji:
                self.merged_emoji_emoji_tag_top_1_total += 1
            else:
                self.merged_word_emoji_tag_top_1_total += 1
        if True in is_res_emoji or True in is_res_emoji_combi:
            self.merged_emoji_tag_top_3_total += 1
            if is_prev_emoji:
                self.merged_emoji_emoji_tag_top_3_total += 1
            else:
                self.merged_word_emoji_tag_top_3_total += 1

        if is_res_emoji_combi[0]:
            self.merged_emoji_tag_combi_top_1_total += 1
            # if is_prev_emoji:
            #     self.merged_emoji_emoji_tag_top_1_total += 1
            # else:
            #     self.merged_word_emoji_tag_top_1_total += 1
        if True in is_res_emoji_combi:
            self.merged_emoji_tag_combi_top_3_total += 1
            # if is_prev_emoji:
            #     self.merged_emoji_emoji_tag_top_3_total += 1
            # else:
            #     self.merged_word_emoji_tag_top_3_total += 1                

        if self.data_util.unk_str == merged_result[0]:
            self.merged_unk_top_1_total += 1
        if self.data_util.unk_str in merged_result:
            self.merged_unk_top_3_total += 1

        if self.data_util.unk_str2 == merged_result[0]:
            self.merged_und_top_1_total += 1
        if self.data_util.unk_str2 in merged_result:
            self.merged_und_top_3_total += 1

        if is_target_emoji:
            if is_res_emoji[0] or is_res_emoji_combi[0]:
                self.merged_emoji_tag_top_1_match += 1
                if is_prev_emoji:
                    self.merged_emoji_emoji_tag_top_1_match += 1
                else:
                    self.merged_word_emoji_tag_top_1_match += 1
            if True in is_res_emoji or True in is_res_emoji_combi:
                self.merged_emoji_tag_top_3_match += 1
                if is_prev_emoji:
                    self.merged_emoji_emoji_tag_top_3_match += 1
                else:
                    self.merged_word_emoji_tag_top_3_match += 1   

            if target_emoji != self.data_util.unk_emoji:
                if target_emoji == merged_result[0]:
                    self.merged_emoji_top_1_count += 1
                    if is_prev_emoji:
                        self.merged_emoji_emoji_top_1_match += 1
                    else:
                        self.merged_word_emoji_top_1_match += 1
                if target_emoji in merged_result:
                    self.merged_emoji_top_3_count += 1
                    if is_prev_emoji:
                        self.merged_emoji_emoji_top_3_match += 1
                    else:
                        self.merged_word_emoji_top_3_match += 1

                contains_target = [(target_emoji in res or res in target_emoji) for res in merged_result]
                if contains_target[0]:
                    self.merged_emoji_combi_top_1_count += 1
                    if is_prev_emoji:
                        self.merged_emoji_emoji_combi_top_1_match += 1
                    else:
                        self.merged_word_emoji_combi_top_1_match += 1
                        print(target_emoji + " ----- " + merged_result[0])
                if True in contains_target:
                    self.merged_emoji_combi_top_3_count += 1
                    if is_prev_emoji:
                        self.merged_emoji_emoji_combi_top_3_match += 1
                    else:
                        self.merged_word_emoji_combi_top_3_match += 1
        else:
            if target_word == self.data_util.unk_str:
                # match unknown
                if target_word == merged_result[0]:
                    self.merged_unk_top_1_match += 1
                if target_word in merged_result:
                    self.merged_unk_top_3_match += 1
            elif target_word == self.data_util.unk_str2:
                # match unknown
                if target_word == merged_result[0]:
                    self.merged_und_top_1_match += 1
                if target_word in merged_result:
                    self.merged_und_top_3_match += 1                    
            else:
                # match word
                if target_word == merged_result[0]:
                    self.merged_word_top_1_count += 1
                if target_word in merged_result:
                    self.merged_word_top_3_count += 1       

    def count_result(self, input_word, raw_target_word, word_result, emoji_result, merged_result):
        self.code_total_count += 1
        is_prev_emoji = self.data_util.is_emoji(input_word)
        is_target_emoji = self.data_util.is_emoji(raw_target_word)

        target_id = self.data_util.outword2id(raw_target_word)
        target_word = self.data_util.id2outword(target_id)

        target_emoji_id = self.data_util.emoji2id(raw_target_word) if is_target_emoji else 0
        target_emoji = self.data_util.id2emoji(target_emoji_id) if is_target_emoji else ""

        if is_target_emoji:
            self.emoji_total_count += 1
            if is_prev_emoji:
                self.emoji_emoji_count += 1
            else:
                self.word_emoji_count += 1
        else:
            self.word_count += 1
            if target_word == self.data_util.unk_str:
                self.unk_word_count += 1
            elif target_word == self.data_util.unk_str2:
                self.und_word_count += 1

        # word and emoji separated result
        self.count_word_result(target_word, word_result, is_prev_emoji, is_target_emoji)
        self.count_emoji_result(target_emoji, emoji_result, is_prev_emoji, is_target_emoji)
        self.count_merged_result(target_word, target_emoji, merged_result, is_prev_emoji, is_target_emoji)

    def print_result(self):
        print('code_total_count \t %d' % self.code_total_count)
        print('emoji_total_count \t %d' % self.emoji_total_count)
        print('word_emoji_count \t %d' % self.word_emoji_count)
        print('emoji_emoji_count \t %d' % self.emoji_emoji_count)
        print('unk_word_count \t %d' % self.unk_word_count)
        # print('und_word_count \t %d' % self.und_word_count)
        print('unk_emoji_count \t %d' % self.unk_emoji_count)
        print('word_count \t %d' % self.word_count)

        print('word_top_1_count \t %d' % self.word_top_1_count)
        print('word_top_3_count \t %d' % self.word_top_3_count)

        print('unk_top_1_match \t %d' % self.unk_top_1_match)
        print('unk_top_3_match \t %d' % self.unk_top_3_match)
        print('unk_top_1_total \t %d' % self.unk_top_1_total)
        print('unk_top_3_total \t %d' % self.unk_top_3_total)

        # print('und_top_1_match \t %d' % self.und_top_1_match)
        # print('und_top_3_match \t %d' % self.und_top_3_match)
        # print('und_top_1_total \t %d' % self.und_top_1_total)
        # print('und_top_3_total \t %d' % self.und_top_3_total)

        print('emoji_top_1_count \t %d' % self.emoji_top_1_count)
        print('emoji_top_3_count \t %d' % self.emoji_top_3_count)        
        print('emoji_tag_top_1_match \t %d' % self.emoji_tag_top_1_match)
        print('emoji_tag_top_3_match \t %d' % self.emoji_tag_top_3_match)
        print('emoji_tag_top_1_total \t %d' % self.emoji_tag_top_1_total)
        print('emoji_tag_top_3_total \t %d' % self.emoji_tag_top_3_total)        

        print('word_emoji_top_1_match \t %d' % self.word_emoji_top_1_match)
        print('word_emoji_top_3_match \t %d' % self.word_emoji_top_3_match)        
        print('word_emoji_tag_top_1_match \t %d' % self.word_emoji_tag_top_1_match)
        print('word_emoji_tag_top_3_match \t %d' % self.word_emoji_tag_top_3_match)
        print('word_emoji_tag_top_1_total \t %d' % self.word_emoji_tag_top_1_total)
        print('word_emoji_tag_top_3_total \t %d' % self.word_emoji_tag_top_3_total)

        print('emoji_emoji_top_1_match \t %d' % self.emoji_emoji_top_1_match)
        print('emoji_emoji_top_3_match \t %d' % self.emoji_emoji_top_3_match)        
        print('emoji_emoji_tag_top_1_match \t %d' % self.emoji_emoji_tag_top_1_match)
        print('emoji_emoji_tag_top_3_match \t %d' % self.emoji_emoji_tag_top_3_match)
        print('emoji_emoji_tag_top_1_total \t %d' % self.emoji_emoji_tag_top_1_total)
        print('emoji_emoji_tag_top_3_total \t %d' % self.emoji_emoji_tag_top_3_total)

        print('emoji_combi_top_1_total \t %d' % self.emoji_combi_top_1_total)
        print('emoji_combi_top_3_total \t %d' % self.emoji_combi_top_3_total)        
        print('emoji_combi_top_1_match \t %d' % self.emoji_combi_top_1_match)
        print('emoji_combi_top_3_match \t %d' % self.emoji_combi_top_3_match)

        print('word_emoji_combi_top_1_total \t %d' % self.word_emoji_combi_top_1_total)
        print('word_emoji_combi_top_3_total \t %d' % self.word_emoji_combi_top_3_total)
        print('word_emoji_combi_top_1_match \t %d' % self.word_emoji_combi_top_1_match)
        print('word_emoji_combi_top_3_match \t %d' % self.word_emoji_combi_top_3_match)
        
        print('emoji_emoji_combi_top_1_total \t %d' % self.emoji_emoji_combi_top_1_total)
        print('emoji_emoji_combi_top_3_total \t %d' % self.emoji_emoji_combi_top_3_total)
        print('emoji_emoji_combi_top_1_match \t %d' % self.emoji_emoji_combi_top_1_match)
        print('emoji_emoji_combi_top_3_match \t %d' % self.emoji_emoji_combi_top_3_match)        
        
        print('merged_word_top_1_count \t %d' % self.merged_word_top_1_count)
        print('merged_word_top_3_count \t %d' % self.merged_word_top_3_count)

        print('merged_unk_top_1_match \t %d' % self.merged_unk_top_1_match)
        print('merged_unk_top_3_match \t %d' % self.merged_unk_top_3_match)
        print('merged_unk_top_1_total \t %d' % self.merged_unk_top_1_total)
        print('merged_unk_top_3_total \t %d' % self.merged_unk_top_3_total)

        # print('merged_und_top_1_match \t %d' % self.merged_und_top_1_match)
        # print('merged_und_top_3_match \t %d' % self.merged_und_top_3_match)
        # print('merged_und_top_1_total \t %d' % self.merged_und_top_1_total)
        # print('merged_und_top_3_total \t %d' % self.merged_und_top_3_total)

        print('merged_emoji_top_1_count \t %d' % self.merged_emoji_top_1_count)
        print('merged_emoji_top_3_count \t %d' % self.merged_emoji_top_3_count)        
        print('merged_emoji_tag_top_1_match \t %d' % self.merged_emoji_tag_top_1_match)
        print('merged_emoji_tag_top_3_match \t %d' % self.merged_emoji_tag_top_3_match)
        print('merged_emoji_tag_top_1_total \t %d' % self.merged_emoji_tag_top_1_total)
        print('merged_emoji_tag_top_3_total \t %d' % self.merged_emoji_tag_top_3_total)

        print('merged_emoji_tag_combi_top_1_total \t %d' % self.merged_emoji_tag_combi_top_1_total)
        print('merged_emoji_tag_combi_top_3_total \t %d' % self.merged_emoji_tag_combi_top_3_total)  

        print('merged_word_emoji_top_1_match \t %d' % self.merged_word_emoji_top_1_match)
        print('merged_word_emoji_top_3_match \t %d' % self.merged_word_emoji_top_3_match)        
        print('merged_word_emoji_tag_top_1_match \t %d' % self.merged_word_emoji_tag_top_1_match)
        print('merged_word_emoji_tag_top_3_match \t %d' % self.merged_word_emoji_tag_top_3_match)
        print('merged_word_emoji_tag_top_1_total \t %d' % self.merged_word_emoji_tag_top_1_total)
        print('merged_word_emoji_tag_top_3_total \t %d' % self.merged_word_emoji_tag_top_3_total)

        print('merged_emoji_emoji_top_1_match \t %d' % self.merged_emoji_emoji_top_1_match)
        print('merged_emoji_emoji_top_3_match \t %d' % self.merged_emoji_emoji_top_3_match)        
        print('merged_emoji_emoji_tag_top_1_match \t %d' % self.merged_emoji_emoji_tag_top_1_match)
        print('merged_emoji_emoji_tag_top_3_match \t %d' % self.merged_emoji_emoji_tag_top_3_match)
        print('merged_emoji_emoji_tag_top_1_total \t %d' % self.merged_emoji_emoji_tag_top_1_total)
        print('merged_emoji_emoji_tag_top_3_total \t %d' % self.merged_emoji_emoji_tag_top_3_total)

        print('merged_emoji_combi_top_1_count \t %d' % self.merged_emoji_combi_top_1_count)
        print('merged_emoji_combi_top_3_count \t %d' % self.merged_emoji_combi_top_3_count)        
        print('merged_emoji_emoji_combi_top_1_match \t %d' % self.merged_emoji_emoji_combi_top_1_match)
        print('merged_emoji_emoji_combi_top_3_match \t %d' % self.merged_emoji_emoji_combi_top_3_match)
        print('merged_word_emoji_combi_top_1_match \t %d' % self.merged_word_emoji_combi_top_1_match)
        print('merged_word_emoji_combi_top_3_match \t %d' % self.merged_word_emoji_combi_top_3_match)

