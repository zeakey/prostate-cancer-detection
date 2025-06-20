find /media/hdd18t/prostate-cancer-with-dce/results_jun19_convert_hdx_interp -mindepth 1 -type d | while read dir; do
    echo "Running evaluation for directory: $dir"
    MAT="${dir}/perpatient.mat"
    if [ -f "$MAT" ]; then
        echo "Skipping $dir, already evaluated."
        continue
    else
        echo "Evaluating $dir..."
        python scripts/measurements_PerPatient.py "$dir" "$MAT"
    fi
    
done