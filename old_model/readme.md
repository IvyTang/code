
使用说明
==============  
本文件夹下旧模型训练的代码

训练
-------------

* 运行  
    `nohup ./train.sh ${data_vocab_path} ${model_save_path} ${graph_save_path} ${config_file} > train.log &`  
    
参数解释：  
`${data_vocab_path} 训练数据id目录 `  
`${model_save_path} 模型参数保存路径 `  
`${graph_save_path} 模型保存路径 `  
`${config_file} 参数配置文件static-sanity-check.cfg, 可自行修改模型参数 `  

测试 
-------------   
* 生成测试结果   
    `nohup python3 input_engine_sparse_new_form.py ${graph_file} ${rnn_vocab_path} ${config_file} ${test_file_in} ${test_file_out} > test.log &`  
    
参数解释：  
`${graph_file} 模型pb文件 `    
`${rnn_vocab_path} 模型词表路径 `  
`${config_file} 参数配置文件 `  
`${test_file_in} 测试数据文件 `  
`${test_file_out} 测试结果文件，可用于计算输入效率、准确率等指标 `  




    


