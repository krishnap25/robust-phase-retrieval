for noise in 1 1e-6 10 
do 
    for cond in 10000 100
	do 
		python main_all.py --cond $cond --noise $noise
		echo "Done $cond $noise"
	done
done
