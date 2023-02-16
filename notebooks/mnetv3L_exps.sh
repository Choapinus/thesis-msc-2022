# mnetv3L
nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv3L --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-5 --optimizer rmsprop --architecture mnetv3L --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &
wait

nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv3L --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-5 --optimizer rmsprop --architecture mnetv3L --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &
wait