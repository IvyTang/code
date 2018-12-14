
使用说明
==============  
本文件夹下为gec英文拼写纠错、单复数/时态等纠错的数据生成、训练代码

训练数据生成
-------------
数据生成的代码：process/data_process.py，lang8原始数据格式见resource/lang8_sample.txt，一行以\t分隔，左边是错句，右边是正确句  
目前只用到正确句子，拼写错误通过map造，单复数/时态错误通过单词lemma形式造平行语料
* 运行  
    `python3 data_process.py

最后在指定的resource/train_data目录下即为生成的训练数据id和词表


训练
-------------
训练的代码在word_correct_model文件夹下
* 运行  
    `nohup ./train.sh ${data_vocab_path} ${model_save_path} ${graph_save_path} ${config_file} > train.log &`  
    
参数解释：  
`${data_vocab_path} 上述生成的训练数据id目录 `  
`${model_save_path} 模型参数保存路径 `  
`${graph_save_path} 模型pb文件保存路径 `  
`${config_file} 参数配置文件static-sanity-check.cfg, 可自行修改模型参数 `  

测试 
-------------  
在test文件夹下 
* 生成测试结果   
    `nohup python3 test.py ${model_file} ${vocab_path} ${config_file} ${test_file_in} ${test_file_out} > test.log &`  
    
参数解释：  
`${model_file} 模型checkpoint路径 `    
`${vocab_path} 模型词表路径 `  
`${config_file} 参数配置文件 `  
`${test_file_in} 测试数据文件，在test_data文件夹里，分为四种错误类型 `  
`${test_file_out} 测试结果文件 `  
* 计算准确率指标  
    `python3 test_analyzer.py ${test_file_native} ${test_file_out}`  

参数解释：  
`${test_file_native} 测试数据的native文件，即正确句子的文件 `  
`${test_file_out} 测试结果文件 `  

模型导出 
-------------  
见export文件夹readme



    


