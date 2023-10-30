# STAN

## step 1
python step1.py --data_dir ../cricket-bowlrelease-dataset --videos videos --features features --batch_size 64

## step 2
### trian 
python step2.py --phase train --model_name LSTM --length_seq 100 --train_stride 10 --test_stride 0 --batch_size 1024 --epoch 100

### test
python step2.py --phase test --model_name LSTM --test_stride 0 --batch_size 1 --resume ./logs/LSTM/2023-10-27\ 23\:52\:11/model_best.pth --save_result

### infer
python step2.py --phase infer --model_name LSTM --infer_stride 0 --batch_size 1 --resume ./logs/LSTM/2023-10-27\ 23\:52\:11/model_best.pth --save_result