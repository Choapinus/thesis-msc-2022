mkdir info

# facial experiments without arcloss
nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv2 > info/out1.ft 2> info/err1.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv3S > info/out2.ft 2> info/err2.ft < /dev/null &
wait
nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv3L > info/out1.ft 2> info/err1.ft < /dev/null &
nohup python train.py --gpu 1 --learning_rate 1e-3 --optimizer sgd --architecture effnet > info/out2.ft 2> info/err2.ft < /dev/null &
wait
nohup python train.py --gpu 0 --learning_rate 1e-4 --optimizer sgd --architecture densenet > info/out1.ft 2> info/err1.ft < /dev/null &

# periocular experiments without arcloss
nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv2 --periocular > info/out2.ft 2> info/err2.ft < /dev/null &
wait
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv3S --periocular > info/out1.ft 2> info/err1.ft < /dev/null &
nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv3L --periocular > info/out2.ft 2> info/err2.ft < /dev/null &
wait
nohup python train.py --gpu 1 --learning_rate 1e-3 --optimizer sgd --architecture effnet --periocular > info/out1.ft 2> info/err1.ft < /dev/null &
nohup python train.py --gpu 0 --learning_rate 1e-4 --optimizer sgd --architecture densenet --periocular > info/out2.ft 2> info/err2.ft < /dev/null &
wait

# facial experiments with arcloss
nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv2 --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv3S --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
wait
nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv3L --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &
nohup python train.py --gpu 1 --learning_rate 1e-3 --optimizer sgd --architecture effnet --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
wait
nohup python train.py --gpu 0 --learning_rate 1e-4 --optimizer sgd --architecture densenet --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &

# periocular experiments with arcloss
nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv2 --periocular --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
wait
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv3S --periocular --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &
nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv3L --periocular --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
wait
nohup python train.py --gpu 1 --learning_rate 1e-3 --optimizer sgd --architecture effnet --periocular --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &
nohup python train.py --gpu 0 --learning_rate 1e-4 --optimizer sgd --architecture densenet --periocular --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
wait