# best exp
nohup python train.py --gpu 0 --multiclass --dataset splunk --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv2 --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --multiclass --dataset splunk --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv2 --arcloss --periocular > info/out1.ft 2> info/err1.ft < /dev/null &
wait

nohup python train.py --gpu 0 --multiclass --dataset splunk --alpha 1.0 --learning_rate 1e-5 --optimizer rmsprop --architecture mnetv2 --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --multiclass --dataset splunk --alpha 1.0 --learning_rate 1e-5 --optimizer rmsprop --architecture mnetv2 --arcloss --periocular > info/out1.ft 2> info/err1.ft < /dev/null &
wait