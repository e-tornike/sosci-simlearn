export CUDA_VISIBLE_DEVICES=$1
export _TYPER_STANDARD_TRACEBACK=1

echo $CUDA_VISIBLE_DEVICES

artifact_dir="/home/tornike/Coding/phd/sosci-simlearn/"

train_path=$artifact_dir"data/meta_pairs_20231221-011820_train.jsonl"
val_path=$artifact_dir"data/meta_pairs_20231221-011820_val.jsonl"
output_dir=$artifact_dir"results"
epochs=50

models="sentence-transformers/distiluse-base-multilingual-cased-v1 sentence-transformers/distiluse-base-multilingual-cased-v2 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 sentence-transformers/paraphrase-multilingual-mpnet-base-v2 intfloat/multilingual-e5-base intfloat/e5-base-v2 thenlper/gte-base T-Systems-onsite/cross-en-de-roberta-sentence-transformer bert-base-multilingual-uncased"
for model in $models; do
    python sosci_simlearn/train.py --train-path=$train_path --val-path=$val_path --output-dir=$output_dir --max-epochs=$epochs --model-name=$model
done