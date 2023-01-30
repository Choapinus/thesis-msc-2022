# facial experiments without arcloss
nohup python train.py --gpu 0 --dataset splunk --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv2 > info/out1.ft 2> info/err1.ft < /dev/null &
# periocular experiments without arcloss
nohup python train.py --gpu 1 --dataset splunk --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv2 --periocular > info/out2.ft 2> info/err2.ft < /dev/null &
wait
# facial experiments with arcloss
nohup python train.py --gpu 0 --dataset splunk --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv2 --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &
# periocular experiments with arcloss
nohup python train.py --gpu 1 --dataset splunk --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv2 --periocular --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
wait