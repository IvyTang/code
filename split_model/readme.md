
使用说明
==============  
本文件夹下为语言模型和键码模型分离的完整代码，支持emoji预测和词组预测.
split_model_with_phrase文件夹下是线上分离模型训练代码，支持词组预测
split_model_with_phrase_emoji_combine文件夹下是分离模型+词组+emoji组合的训练代码  
两套代码基本一致，用法以split_model_with_phrase下的线上分离模型训练代码为例进行说明  

训练数据生成
-------------
数据生成的代码：process/data_process.py
* 运行  
    `nohup python3 data_process.py ${words_dict_file} ${words_map_file} ${emojis_file} ${phrase_file} ${data_path_in} ${data_path_out} ${rate_threshold} ${words_num} ${phrase_num} ${train_num} ${dev_num} ${test_num} > out.log &`  

参数解释：  
`${words_dict_file} 16万大词表文件 `  
`${words_map_file} 错误键码map文件 `  
`${emojis_file} emoji词表 `  
`${phrase_file} 词组词表 `  
`${data_path_in} 原始训练数据目录 `  
`${data_path_out} 生成的训练数据id目录 `  
`${rate_threshold} 筛数据时的阈值，一般为0.8 `  
`${words_num} 用于训练的词表大小，一般为20000 `  
`${phrase_num} 用于训练的词组词表大小，一般为2000 `    
`${train_num} 用于train的完整句子数，一般取300万到500万为宜 `  
`${dev_num} 用于dev的完整句子数，一般为1万 `  
`${test_num} 用于test的完整句子数，一般为1万 `  

最后在指定的${data_path_out}目录下即为生成的训练数据id和词表  


训练
-------------
训练的代码在split_model里
* 运行  
    `nohup ./train.sh ${data_vocab_path} ${model_save_path} ${graph_save_path} ${config_file} > train.log &`  
    
参数解释：  
`${data_vocab_path} 上述生成的训练数据id目录 `  
`${model_save_path} 模型参数保存路径 `  
`${graph_save_path} 模型保存路径,最后会保存三种模型,即后缀分别为lm、kc_full和kc_slim的模型，测试时候只需要用到kc_slim模型 `  
`${config_file} 参数配置文件static-sanity-check.cfg, 可自行修改模型参数 `  

测试 
-------------   
* 生成测试结果   
    `nohup python3 test.py ${graph_file} ${rnn_vocab_path} ${config_file} ${test_file_in} ${test_file_out} > test.log &`  
    
参数解释：  
`${graph_file} 模型文件，即保存的kc_slim模型 `    
`${rnn_vocab_path} rnn词表路径 `  
`${config_file} 参数配置文件 `  
`${test_file_in} 测试数据文件, 在resource/test_data下 `  
`${test_file_out} 测试结果文件，可用于计算输入效率、准确率等指标 `  
* 计算指标  
    `python3 metrics_analyzer.py ${test_file_out} ${rnn_vocab_file} ${full_vocab_file} ${emojis_file} ${phrase_file}`  

参数解释：  
`${test_file_out} 测试结果文件 `    
`${rnn_vocab_file} rnn词表 `  
`${full_vocab_file} 16万大词表 `  
`${emojis_file} emoji词表 `  
`${phrase_file} 词组词表 ` 



    


