
for i in {8..14}
do
    python space_carve.py -n $i
    python compute_K.py -n $i

    python demo_deform.py -n $i
done