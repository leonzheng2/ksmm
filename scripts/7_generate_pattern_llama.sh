emb_dim=4096
ffn_dim=14336

# up projection
python src/ksmm_py/pattern/generate_patterns.py \
  --output-dim $ffn_dim \
  --input-dim $emb_dim \
  --ratio-rewrite-multiply 0.01 \
  --density 0.25 \
  --power-list 1 5 1 1 5 1 \
  --save-dir "results/patterns" \
  --comma-separator

# down projection
python src/ksmm_py/pattern/generate_patterns.py \
  --output-dim $emb_dim \
  --input-dim $ffn_dim \
  --ratio-rewrite-multiply 0.01 \
  --density 0.25 \
  --power-list 1 5 1 1 5 1 \
  --save-dir "results/patterns" \
  --comma-separator
