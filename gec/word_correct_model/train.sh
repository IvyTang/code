#! /bin/bash

data_vocab_path=$1
save_path=$2
graph_save_path=$3
config=$4

python3 main.py --phase=lm --save_path=${save_path} --graph_save_path=${graph_save_path} --data_path=${data_vocab_path} --vocab_path=${data_vocab_path} --model_config=${config}
python3 main.py --phase=letter --save_path=${save_path} --graph_save_path=${graph_save_path} --data_path=${data_vocab_path} --vocab_path=${data_vocab_path} --model_config=${config}
python3 main.py --phase=softmax --save_path=${save_path} --graph_save_path=${graph_save_path} --data_path=${data_vocab_path} --vocab_path=${data_vocab_path} --model_config=${config}
