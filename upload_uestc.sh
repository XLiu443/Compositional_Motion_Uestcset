while :
do
    gdrive list -q "'1u_5fuUZ8r1Upffb68Mfmivgw7sWXlSQh' in parents" --no-header --max 0 | cut -d" " -f1 - | xargs -L 1 gdrive delete
    gdrive upload --parent 1u_5fuUZ8r1Upffb68Mfmivgw7sWXlSQh /fsx/sernamlim/xiaoliu/Compositional_Motion_Uestcset/main/Compositional_Motion_Uestcset/slurm_main.out
    echo "Sleeping for 120"
    sleep "120"
done
