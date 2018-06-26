for cond in 1 100 10000 
do 
	for noise in 1 1e-6 10 
	do 
		python main_all.py --cond $cond --noise $noise
		echo "Done $cond $noise"
	done
done
