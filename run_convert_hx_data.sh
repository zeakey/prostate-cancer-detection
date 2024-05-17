set -ex

for method in AttentionUNet3D  nnUNet  Proposed_AtPCaNet  ResidualUNet3D  SEResUNet3D   VNet  VoxResNet #unetr
do
    evals=()
    for seed in 0 1 2 3
    do
        for scale in 0.01 0.02 0.03 0.04 0.06 0.08 0.1
        do
            savedir="/media/hdd2/prostate-cancer-with-dce/trick/may-17/$method/seed-$seed-randscale-$scale"
            perpatient_path="$savedir/Proposed_AtPCaNet.perpatient.mat"
            if [ ! -f $perpatient_path ]; then
                    python scripts/convert_haoxin_data.py --method $method --seed $seed --save-dir $savedir --randscale $scale
                    python evaluate.py "$savedir/$method/FN" "$savedir/$method.mat"
                    python scripts/measurements_PerPatient.py "$savedir/$method/all" "$savedir/$method.perpatient.mat"
            fi
            evals+=( "$savedir/$method.mat" )
        done
    done
    python  draw_roc.py --evals ${evals[@]} --savefig "/media/hdd2/prostate-cancer-with-dce/trick/may-17/$method-roc.pdf"
done

# python  draw_roc.py --evals ${evals[@]} --savefig april14-FN.pdf
# python scripts/convert_haoxin_data.py --seed 1 --save-dir /media/hdd2/prostate-cancer-with-dce/trick/1