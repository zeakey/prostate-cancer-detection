set -ex

SAVE_DIR=/webdata/prostate/cancer_detection_crosslice/results_PICAI_kai_oct4

mkdir -p $SAVE_DIR

# python scripts/convert_picai_data.py --savedir ${SAVE_DIR} --alpha 0 --beta 0 --seed 0 --debug
for seed in 0 1 2; do
    for alpha in 0.1 0.2 0.4 0.8 1.6; do
        for beta in 0.1 0.2 0.4 0.8; do
            python scripts/convert_picai_data.py --savedir ${SAVE_DIR} --rand --alpha $alpha --beta $beta --seed $seed --debug 
        done
    done
done

cd "$(dirname "$SAVE_DIR")" && tar -cf "$SAVE_DIR.tar" "$(basename "$SAVE_DIR")"
scp "$SAVE_DIR.tar" alex@stellaris:/media/hdd1/alex/cancer_detection_crosslice/