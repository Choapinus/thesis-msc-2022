run:
	python main.py

clean:
	rm -rf *.pyc

setup:
	# define a setup, use conda, env.yml and then create a setup.py.
	# it will be used to install the dependencies into future
	# python setup.py build
	# python setup.py install