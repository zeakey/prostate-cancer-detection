set -e

savedir="/media/hdd18t/prostate-cancer-with-dce/results_PICAI_20jun19convert_interp"

METHODS="VTUNet,ResidualUNet,3DUNet,FocalNet"

for seed in $(seq 1 1 5); do
    for alpha in $(seq 1 2 8); do
        for beta in $(seq 0.7 0.1 1.2); do
            savedir1="${savedir}/alpha_${alpha}_beta_${beta}_seed_${seed}"
            echo "Running with alpha=${alpha} beta=${beta} and seed=${seed}, saving to ${savedir1}."
            python scripts/convert_picai_data3.py --methods ${METHODS} --alpha ${alpha} --beta ${beta} --seed ${seed} --savedir ${savedir1}
            python measurement_picai.py ${savedir1}
        done
    done
    
done
