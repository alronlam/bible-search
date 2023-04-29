.PHONY: help conda-env setup requirements
-include .env

conda-env:
	conda env create --prefix ./env -f environment.yml --no-default-packages

setup:
	pip install pip-tools==6.13.*
	pip-sync requirements.txt

requirements:
	pip-compile requirements.in -o requirements.txt -v
	pip-sync requirements.txt
