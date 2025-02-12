if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LookBackWindow" ]; then
    mkdir ./logs/LookBackWindow
fi

pred_len=720
model_name=FEDformer

for seq_len in 48 72 96 120 144 168 192 336 504 672 720
do
  python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path electricity.csv \
      --model_id electricity_$seq_len_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --itr 1 >logs/LookBackWindow/$model_name'_electricity'_$seq_len'_'$pred_len.log
done
