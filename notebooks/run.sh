mkdir -p info/outs info/errs

python train.py --gpu 0 --epochs 2 > info/out_00.ft 2> info/err_00.ft < /dev/null &
python train.py --gpu 1 --epochs 2 > info/out_01.ft 2> info/err_01.ft < /dev/null &
wait -n
python train.py --gpu 0 --epochs 2 > info/out_02.ft 2> info/err_02.ft < /dev/null &
python train.py --gpu 1 --epochs 2 > info/out_03.ft 2> info/err_03.ft < /dev/null &
wait