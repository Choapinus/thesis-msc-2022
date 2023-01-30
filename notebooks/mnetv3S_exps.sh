nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-5 --optimizer rmsprop --architecture mnetv3S > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-5 --optimizer rmsprop --architecture mnetv3S --periocular > info/out1.ft 2> info/err1.ft < /dev/null &
wait
nohup python train.py --gpu0 --alpha 1.0 --learning_rate 1e-5 --optimizer rmsprop --architecture mnetv3S --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-5 --optimizer rmsprop --architecture mnetv3S --periocular --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &
wait

# adam opt
nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer adam --architecture mnetv3S > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-4 --optimizer adam --architecture mnetv3S --periocular > info/out1.ft 2> info/err1.ft < /dev/null &
wait
nohup python train.py --gpu0 --alpha 1.0 --learning_rate 1e-4 --optimizer adam --architecture mnetv3S --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-4 --optimizer adam --architecture mnetv3S --periocular --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &
wait

nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-5 --optimizer adam --architecture mnetv3S > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-5 --optimizer adam --architecture mnetv3S --periocular > info/out1.ft 2> info/err1.ft < /dev/null &
wait
nohup python train.py --gpu0 --alpha 1.0 --learning_rate 1e-5 --optimizer adam --architecture mnetv3S --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-5 --optimizer adam --architecture mnetv3S --periocular --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &
wait

# sgd opt
nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-4 --optimizer sgd --architecture mnetv3S > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-4 --optimizer sgd --architecture mnetv3S --periocular > info/out1.ft 2> info/err1.ft < /dev/null &
wait
nohup python train.py --gpu0 --alpha 1.0 --learning_rate 1e-4 --optimizer sgd --architecture mnetv3S --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-4 --optimizer sgd --architecture mnetv3S --periocular --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &
wait

nohup python train.py --gpu 0 --alpha 1.0 --learning_rate 1e-5 --optimizer sgd --architecture mnetv3S > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-5 --optimizer sgd --architecture mnetv3S --periocular > info/out1.ft 2> info/err1.ft < /dev/null &
wait
nohup python train.py --gpu0 --alpha 1.0 --learning_rate 1e-5 --optimizer sgd --architecture mnetv3S --arcloss > info/out2.ft 2> info/err2.ft < /dev/null &
nohup python train.py --gpu 1 --alpha 1.0 --learning_rate 1e-5 --optimizer sgd --architecture mnetv3S --periocular --arcloss > info/out1.ft 2> info/err1.ft < /dev/null &
wait