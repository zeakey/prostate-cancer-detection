set -ex

if [ $# -lt 1 ]; then
    echo "at least one arg (got $#)"
    exit 1
else
    echo $#
    savedir=$1
fi

for method in Proposed_AtPCaNet  #ResidualUNet3D  SEResUNet3D   VNet  VoxResNet unetr AttentionUNet3D  nnUNet
do
    evals_all=()
    evals_fn=()
    evals_patient=()
    savedir1="${savedir}/$method/origin"
    python scripts/convert_haoxin_data.py --method $method --save-dir $savedir1
    python evaluate.py "$savedir1/$method/all" "$savedir1/eval.all.mat"
    python evaluate.py "$savedir1/$method/FN" "$savedir1/eval.fn.mat"
    python evaluate.py "$savedir1/$method/TP" "$savedir1/eval.tp.mat"
    python evaluate.py "$savedir1/$method/small" "$savedir1/eval.small.mat"
    python evaluate.py "$savedir1/$method/large" "$savedir1/eval.large.mat"
    python evaluate.py "$savedir1/$method/PZ" "$savedir1/eval.pz.mat"
    python evaluate.py "$savedir1/$method/TZ" "$savedir1/eval.tz.mat"
    #
    find $savedir1 -name "*_pred.npy" -exec rm {} \;
    find $savedir1 -name "*_mask.npy" -exec rm {} \;

    evals_all+=( $all)
    evals_fn+=( $fn)
    evals_patient+=($patient)
    for seed in 0 1 2 3
    do
        for beta in 0 0.5 1
        do
            for gamma in -1 -0.125 0 0.125 0.5 1
            do
                savedir1="$savedir/$method/alpha$alpha-beta$beta-gamma$gamma"
                #
                all="$savedir1/eval.all.mat"
                fn="$savedir1/eval.fn.mat"
                tp="$savedir1/eval.tp.mat"
                small="$savedir1/eval.small.mat"
                large="$savedir1/eval.large.mat"
                patient="$savedir1/eval.perpatient.mat"
                #
                if [[ ! -e "$all" || ! -e "$fn" || ! -e "$patient" ]]; then
                    python scripts/convert_haoxin_data.py --method $method --rand --alpha $alpha --beta $beta --gamma $gamma --save-dir $savedir1
                    python evaluate.py "$savedir1/$method/all" $all
                    python evaluate.py "$savedir1/$method/FN" $fn
                    python evaluate.py "$savedir1/$method/TP" $tp
                    python evaluate.py "$savedir1/$method/PZ" "$savedir1/eval.pz.mat"
                    python evaluate.py "$savedir1/$method/TZ" "$savedir1/eval.tz.mat"
                    # python evaluate.py "$savedir1/$method/small" $small
                    # python evaluate.py "$savedir1/$method/large" $large
                    # python scripts/measurements_PerPatient.py "$savedir1/$method/all" $patient
                    find $savedir1 -name "*_pred.npy" -exec rm {} \;
                    find $savedir1 -name "*_mask.npy" -exec rm {} \;
                fi
                evals_all+=( $all)
                evals_fn+=( $fn)
                evals_patient+=($patient)
            done
        done
    done
    # python  draw_roc.py --evals ${evals_all[@]} --savefig "$savedir/$method-all-roc.pdf"
    # python  draw_roc.py --evals ${evals_fn[@]} --savefig "$savedir/$method-fn-roc.pdf"
    # python  draw_roc.py --evals ${evals_patient[@]} --savefig "$savedir/$method-patient-roc.pdf"
done

# python  draw_roc.py --evals ${evals[@]} --savefig april14-FN.pdf
# python scripts/convert_haoxin_data.py --seed 1 --save-dir /media/hdd2/prostate-cancer-with-dce/trick/1
