# whetstone
```
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=\$gpu nohup .venv/bin/python scripts/harvest.py \
    --input  data/interim/<POOL>.jsonl \
    --output data/raw/<RUN>_harvest_w\${gpu}.jsonl \
    --model  google/gemma-4-E4B-it \
    --K 4 --temperature 0.9 --top_p 0.95 \
    --max_tokens 32000 --max_model_len 33024 \
    --tp 1 --gpu_mem 0.85 \
    --worker_id \$gpu --n_workers 8 --batch 64 \
    > logs/<RUN>_harvest_w\${gpu}.log 2>&1 &
done
```
