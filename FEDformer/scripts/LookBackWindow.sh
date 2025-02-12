# cd FEDformer
if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LookBackWindow" ]; then
    mkdir ../logs/LookBackWindow
fi

pred_len=720
for seqLen in 24 48 72 96 120 144 168 192 336 504 672 720
do
## electricity
python -u run.py \
 --is_training 1 \
 --root_path .../dataset/ \
 --data_path electricity.csv \
 --task_id ECL \
 --model FEDformer \
 --data custom \
 --features M \
 --seq_len $seqLen \
 --label_len 48 \
 --pred_len $pred_len \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 321 \
 --dec_in 321 \
 --c_out 321 \
 --des 'Exp' \
 --itr 1 >../logs/LookBackWindow/FEDformer_electricity_$seqLen'_'$pred_len.log
done
# cd ..
