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
    #
    python scripts/measurements_PerPatient.py "$savedir1/$method/all" "$savedir1/eval.perpatient.mat"
    python evaluate.py "$savedir1/$method/all" "$savedir1/eval.all.mat"
    python evaluate.py "$savedir1/$method/FN" "$savedir1/eval.fn.mat"
    python evaluate.py "$savedir1/$method/TP" "$savedir1/eval.tp.mat"
    python evaluate.py "$savedir1/$method/small" "$savedir1/eval.small.mat"
    python evaluate.py "$savedir1/$method/large" "$savedir1/eval.large.mat"
    python evaluate.py "$savedir1/$method/PZ" "$savedir1/eval.pz.mat"
    python evaluate.py "$savedir1/$method/TZ" "$savedir1/eval.tz.mat"
    # #
    # find $savedir1 -name "*_pred.npy" -exec rm {} \;
    # find $savedir1 -name "*_mask.npy" -exec rm {} \;

    evals_all+=( $all)
    evals_fn+=( $fn)
    evals_patient+=($patient)
    for seed in 0
    do
        for beta in 0
        do
            for gamma in 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25
            do
                savedir1="$savedir/$method/seed$seed-beta$beta-gamma$gamma"
                #
                if [[ ! -e "$all" || ! -e "$fn" || ! -e "$patient" ]]; then
                    python scripts/convert_haoxin_data.py --method $method --rand --beta $beta --gamma $gamma --save-dir $savedir1
                    python evaluate.py "$savedir1/$method/all" "$savedir1/eval.all.mat"
                    python evaluate.py "$savedir1/$method/FN" "$savedir1/eval.fn.mat"
                    python evaluate.py "$savedir1/$method/TP" "$savedir1/eval.tp.mat"
                    python evaluate.py "$savedir1/$method/PZ" "$savedir1/eval.pz.mat"
                    python evaluate.py "$savedir1/$method/TZ" "$savedir1/eval.tz.mat"
                    python evaluate.py "$savedir1/$method/small" "$savedir1/eval.small.mat"
                    python evaluate.py "$savedir1/$method/large" "$savedir1/eval.large.mat"
                    python scripts/measurements_PerPatient.py "$savedir1/$method/all" "$savedir1/eval.perpatient.mat"
                    find $savedir1 -name "*_pred.npy" -exec rm {} \;
                    find $savedir1 -name "*_mask.npy" -exec rm {} \;
                fi
                evals_all+=( $all)
                evals_fn+=( $fn)
                evals_patient+=($patient)
            done
        done
    done
done
