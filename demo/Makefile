env: venv/touchfile

# TODO add milvus index creating example code

venv/touchfile: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/touchfile

router: env
	. venv/bin/activate; streamlit run router_demo.py

clean:
	rm -rf venv
	find -iname "*.pyc" -delete
