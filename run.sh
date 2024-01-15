export CUDA_VISIBLE_DEVICES=$1
export _TYPER_STANDARD_TRACEBACK=1

echo $CUDA_VISIBLE_DEVICES

timestamp=$(date +'%m-%d-%Y_%H-%M')

artifact_dir="/home/tornike/Coding/phd/sosci-simlearn/"
train_path=$artifact_dir"data/filtered_meta_pairs_20240113-122949_train.jsonl"
val_path=$artifact_dir"data/filtered_meta_pairs_20240113-122949_val.jsonl"
log_dir=$artifact_dir
cache_folder=""

epochs=50
samples="200 2000 20000 200000"
seeds="42 1234 1337"

for sample in $samples; do
    for seed in $seeds; do
        output_dir=$artifact_dir"results/filtered/"$timestamp"/sample="$sample"/seed="$seed

        models="sentence-transformers/distiluse-base-multilingual-cased-v1 sentence-transformers/distiluse-base-multilingual-cased-v2 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 sentence-transformers/paraphrase-multilingual-mpnet-base-v2 intfloat/multilingual-e5-base intfloat/e5-base-v2 thenlper/gte-base T-Systems-onsite/cross-en-de-roberta-sentence-transformer bert-base-multilingual-uncased"
        for model in $models; do
            python sosci_simlearn/train.py --train-path=$train_path --val-path=$val_path --output-dir=$output_dir --max-epochs=$epochs --model-name=$model --sample-n=$sample --seed=$seed --log-dir=$log_dir --cache-folder=$cache_folder
        done
    done
done