export-req:
	pip freeze > req.txt

install:
	pip install -r req.txt

