# mnetv3L
# nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv3L --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
# wait

# mnetv2
nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv2 > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv2 --periocular > info/out1.ft 2> info/err1.ft < /dev/null &
wait
nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv2 --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture mnetv2 --periocular --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &
wait

# densenet
nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture densenet > info/out1.ft 2> info/err1.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture densenet --periocular > info/out2.ft 2> info/err2.ft < /dev/null &
wait

nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture densenet --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-4 --optimizer rmsprop --architecture densenet --arcloss --periocular > info/out2.ft 2> info/err2.ft < /dev/null &
wait

nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer adam --architecture densenet > info/out1.ft 2> info/err1.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-4 --optimizer adam --architecture densenet --periocular > info/out2.ft 2> info/err2.ft < /dev/null &
wait

nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer adam --architecture densenet --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-4 --optimizer adam --architecture densenet --arcloss --periocular > info/out2.ft 2> info/err2.ft < /dev/null &
wait