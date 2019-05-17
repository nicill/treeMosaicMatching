rm ./wm1/patches/*.*
rm ./wm1/wMAll.csv

for i in 1 2 3
do
	rm ./wm1/wM$i.csv
	echo "	python imagePatcherAnnotator.py ./wm1/wM$i 250 ./wm1/patches wM$i "
	python imagePatcherAnnotator.py ./wm1/wM$i 250 ./wm1/patches wM$i >> out.txt
	cat ./wm1/wM$i.csv >> ./wm1/wMAll.csv
done
