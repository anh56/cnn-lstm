export-req:
	pip freeze > req.txt

install:
	pip install -r req.txt

test:
	python main.py --cvss_col access_vector

sa:
	python summarize.py --all

sc:
	python summarize.py --cvss_col $(c)

sa:
	python summarize.py --arch $(c)