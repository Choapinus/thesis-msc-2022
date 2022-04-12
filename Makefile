run:
	# python main.py
	docker-compose up

clean:
	rm -rf *.pyc
	rm -rf __pycache__
	docker-compose down

setup:
	# define a setup, use conda, env.yml and then create a setup.py.
	# it will be used to install the dependencies into future
	# python setup.py build
	# python setup.py install