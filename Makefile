export-req:
	pip freeze > req.txt

install:
	pip install -r req.txt

test:
	python main.py --cvss_col access_vector
