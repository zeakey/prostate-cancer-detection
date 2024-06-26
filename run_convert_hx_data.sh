set -ex

savedir="/media/hdd2/prostate-cancer-with-dce/trick/may-21test"

for method in Proposed_AtPCaNet  #ResidualUNet3D  SEResUNet3D   VNet  VoxResNet unetr AttentionUNet3D  nnUNet
do
    evals_all=()
    evals_fn=()
    evals_patient=()
    savedir1="${savedir}/$method/origin"
    #
    all="$savedir1/eval.all.mat"
    fn="$savedir1/eval.falsenegative.mat"
    patient="$savedir1/eval.perpatient.mat"

    if [[ ! -e "$all" || ! -e "$fn" || ! -e "$patient" ]]; then
        python scripts/convert_haoxin_data.py --method $method --save-dir $savedir1
        python evaluate.py "$savedir1/$method/all" $all
        python evaluate.py "$savedir1/$method/FN" $fn
        python scripts/measurements_PerPatient.py "$savedir1/$method/all" $patient
    fi
    evals_all+=( $all)
    evals_fn+=( $fn)
    evals_patient+=($patient)
    for seed in 0 1 2
    do
        for alpha in 2 4 8
        do
            savedir1="${savedir}/$method/seed-$seed-alpha-$alpha"
            #
            all="$savedir1/eval.all.mat"
            fn="$savedir1/eval.falsenegative.mat"
            patient="$savedir1/eval.perpatient.mat"
            #
            if [[ ! -e "$all" || ! -e "$fn" || ! -e "$patient" ]]; then
                python scripts/convert_haoxin_data.py --method $method --rand --seed $seed --alpha $alpha --save-dir $savedir1
                python evaluate.py "$savedir1/$method/all" $all
                python evaluate.py "$savedir1/$method/FN" $fn
                python scripts/measurements_PerPatient.py "$savedir1/$method/all" $patient
            fi
            evals_all+=( $all)
            evals_fn+=( $fn)
            evals_patient+=($patient)
        done
    done
    python  draw_roc.py --evals ${evals_all[@]} --savefig "$savedir/$method-all-roc.pdf"
    python  draw_roc.py --evals ${evals_fn[@]} --savefig "$savedir/$method-fn-roc.pdf"
    python  draw_roc.py --evals ${evals_patient[@]} --savefig "$savedir/$method-patient-roc.pdf"
done

# python  draw_roc.py --evals ${evals[@]} --savefig april14-FN.pdf
# python scripts/convert_haoxin_data.py --seed 1 --save-dir /media/hdd2/prostate-cancer-with-dce/trick/1
