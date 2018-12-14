
使用说明
==============
用于tf_serving的模型保存在model_for_tf_serving下

例如,输入一句话为"I love you",则输入格式为
* 语言模型输入
lm_input: [id(bos),id(I),id(love),id(you),id(eos),id(pad),....],长度为80
lm_length: [words_num],即输入单词数(不包括bos和eos,本例中为3). 注意, 该参数不可省略, 否则会影响语言模型最后输入到键码模型的状态的正确性
lemma_input： [id(lemma(I)),id(lemma(love)),....],长度为80

* 键码模型输入
kc_input: [[id(start),id(i),id(pad),id(pad),id(pad)],[id(start),id(l),id(o),id(v),id(e)],[id(start),id(y),id(o),id(u),id(pad)],[id(pad)...]...],是一个二维矩阵, 维度为[80,max_word_length+1], 在本例中即为[80,5], 超过单词数的行, 即从第4行起全部补零
kc_length: [1,4,3,0,0,.....], 是一个一维矩阵, 维度为[1，80], 依次表示"I love you"的各个单词长度(不包括start_tag), 超过单词数后补0

输出格式为
* top_k_id输出:
output_values: 维度为[words_num, 3]的矩阵, 表示每个输入单词对应的top3输出单词id
* top_k_prob输出:
output_values: 维度为[words_num, 3]的矩阵, 表示每个输入单词对应的top3输出单词的概率
* lemma_top_k_id输出:
lemma_output_values: 维度为[words_num, 3]的矩阵, 表示每个输入单词对应的lemma模型输出的top3输出单词id
* lemma_top_k_prob输出:
lemma_output_values: 维度为[words_num, 3]的矩阵, 表示每个输入单词对应的lemma模型输出top3输出单词的概率

在本机测试效果
-------------   
* 执行：    
    `python3 test_checkpoint.py ${model_file} ${vocab_path} ${config_file} ${export_path}`

参数解释：  
`${model_file} 模型checkpoint文件 `    
`${vocab_path} 模型词表路径 `  
`${config_file} 参数配置文件 `  
`${export_path} 导出路径 `  


    


