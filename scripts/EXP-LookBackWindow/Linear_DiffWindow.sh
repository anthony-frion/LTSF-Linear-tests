if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LookBackWindow" ]; then
    mkdir ./logs/LookBackWindow
fi

model_name=DLinear
pred_len=720

for seq_len in 48 72 96 120 144 168 192 336 504 672 720
do
   python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id Electricity_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len  \
    --enc_in 321 \
    --des 'Exp' \
    --itr 1 --batch_size 16  --learning_rate 0.001 >logs/LookBackWindow/$model_name'_'electricity_$seq_len'_'$pred_len.log
done
