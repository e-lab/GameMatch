# try to run experiments
# failing
# do not use this to run

for (( i = 0; i <= 5; i++ ))
do
    echo "running test $i"
    
    let j=0
    for l in 16 32 64 128
    do
        echo "Combo: $l $j"
        python3 experiment.py -l $l -d $j &
        let j+=1 
    done
done
