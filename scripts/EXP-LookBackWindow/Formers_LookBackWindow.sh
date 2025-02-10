if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LookBackWindow" ]; then
    mkdir ./logs/LookBackWindow
fi
pred_len=720

for model_name in Autoformer Informer Transformer
do
for seq_len in 24 48 72 96 120 144 168 192 336 504 672 720
do
  python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path electricity.csv \
      --model_id electricity_96_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
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
done
