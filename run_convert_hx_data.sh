set -ex

for method in Proposed_AtPCaNet  #ResidualUNet3D  SEResUNet3D   VNet  VoxResNet unetr AttentionUNet3D  nnUNet
do
    evals_all=()
    evals_fn=()
    evals_patient=()
    for seed in 0 1 2
    do
        for scale in 0.01 0.02 0.03 0.04 0.06 # 0.08 0.1
        do
            savedir="/media/hdd2/prostate-cancer-with-dce/trick/may-17b/$method/seed-$seed-randscale-$scale"
            #
            all="$savedir/$method.all.mat"
            fn="$savedir/$method.falsenegative.mat"
            patient="$savedir/$method.perpatient.mat"
            #
            if [[ ! -f $all || ! -f $fn || -f $patient ]]; then
                    python scripts/convert_haoxin_data.py --method $method --seed $seed --randscale $scale --save-dir $savedir
                    python evaluate.py "$savedir/$method/all" $all
                    python evaluate.py "$savedir/$method/FN" $fn
                    python scripts/measurements_PerPatient.py "$savedir/$method/all" $patient
            fi
            evals_all+=( $all)
            evals_fn+=( $fn)
            evals_patient+=($patient)
        done
    done
    python  draw_roc.py --evals ${evals_all[@]} --savefig "/media/hdd2/prostate-cancer-with-dce/trick/may-17/$method-all-roc.pdf"
    python  draw_roc.py --evals ${evals_fn[@]} --savefig "/media/hdd2/prostate-cancer-with-dce/trick/may-17/$method-fn-roc.pdf"
    python  draw_roc.py --evals ${evals_patient[@]} --savefig "/media/hdd2/prostate-cancer-with-dce/trick/may-17/$method-patient-roc.pdf"
done

# python  draw_roc.py --evals ${evals[@]} --savefig april14-FN.pdf
# python scripts/convert_haoxin_data.py --seed 1 --save-dir /media/hdd2/prostate-cancer-with-dce/trick/1