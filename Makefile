SHELL := /bin/bash
.DEFAULT_GOAL := default
.PHONY: \
	help default all \
	clean clean-build clean-pyc clean-test \
	clean-$(VENV_NAME) purge \
	install install-all \
	install-requirements install-dev-requirements \
	docker-build test test-docker lint flake8 pylint isort

VENV_INTERP = python3.7
VENV_NAME ?= venv

PIP_INST_EXTRA = 
PIP_INST_DEV = development
PIP_INST_DEMO = demo
PIP_INST_ALL = $(PIP_INST_DEV),$(PIP_INST_DEMO)
PIP_FLAG = 
PIP_INST_FLAG = 
PIP_INST_EDIT = -e

REQUIREMENTS_FILE = requirements.txt
DEV_REQUIREMENTS_FILE = requirements.dev.txt

DOCKER_TAG = sertansenturk/makam-recognition-experiments
DOCKER_VER = latest
DOCKER_FILE = Dockerfile
DOCKER_FLAG = 

HELP_PADDING = 28
bold := $(shell tput bold)
sgr0 := $(shell tput sgr0)
padded_str := %-$(HELP_PADDING)s
pretty_command := $(bold)$(padded_str)$(sgr0)

help:
	@printf "======= General ======\n"
	@printf "$(pretty_command): run \"default\" (see below)\n"
	@printf "$(pretty_command): run \"purge\", \"$(VENV_NAME)\", and \"install\"\n" default
	@printf "$(pretty_command): run \"purge\", \"$(VENV_NAME)\", and \"install-dev-requirements\"\n" dev
	@printf "$(pretty_command): run \"purge\", \"$(VENV_NAME)\", and \"install-all\"\n" all
	@printf "\n"
	@printf "======= Cleanup ======\n"
	@printf "$(pretty_command): remove all build, test, coverage and python artifacts\n" clean
	@printf "$(pretty_command): \"clean\" all above and remove the virtualenv\n" purge
	@printf "$(padded_str)VENV_NAME, virtualenv name (default: $(VENV_NAME))\n"
	@printf "$(pretty_command): remove build artifacts\n" clean-build
	@printf "$(pretty_command): remove python file artifacts\n" clean-pyc
	@printf "$(pretty_command): remove test artifacts\n" clean-test
	@printf "$(pretty_command): remove python virtualenv\n" clean-$(VENV_NAME)
	@printf "$(padded_str)VENV_NAME, virtualenv name (default: $(VENV_NAME))\n"
	@printf "\n"
	@printf "======= Setup =======\n"
	@printf "$(pretty_command): create a virtualenv\n" $(VENV_NAME)
	@printf "$(padded_str)VENV_NAME, virtualenv name (default: $(VENV_NAME))\n"
	@printf "$(padded_str)VENV_INTERP, python interpreter (default: $(VENV_INTERP))\n"
	@printf "$(pretty_command): install in a virtualenv\n" install
	@printf "$(padded_str)VENV_NAME, virtualenv name to install (default: $(VENV_NAME))\n"
	@printf "$(padded_str)PIP_FLAG, pip flags (default: $(PIP_FLAG))\n"
	@printf "$(padded_str)PIP_INST_FLAG, pip install flags (default: $(PIP_INST_FLAG))\n"
	@printf "$(pretty_command): install in editable mode and in a virtualenv with all extra dependencies\n" install-all
	@printf "$(padded_str)VENV_NAME, virtualenv name to install (default: $(VENV_NAME))\n"
	@printf "$(padded_str)PIP_FLAG, pip flags (default: $(PIP_FLAG))\n"
	@printf "$(pretty_command): install libraries in $(REQUIREMENTS_FILE) to the virtualenv\n" install-requirements
	@printf "$(padded_str)VENV_NAME, virtualenv name to install (default: $(VENV_NAME))\n"
	@printf "$(pretty_command): install development libraries in $(DEV_REQUIREMENTS_FILE) to the virtualenv\n" install-dev-requirements
	@printf "$(padded_str)VENV_NAME, development virtualenv name to install (default: $(VENV_NAME))\n"
	@printf "\n"
	@printf "======= Docker =======\n"
	@printf "$(pretty_command): build docker image\n" docker-build
	@printf "$(padded_str)DOCKER_TAG, docker image tag (default: $(DOCKER_TAG))\n"
	@printf "$(padded_str)DOCKER_VER, docker image version (default: $(DOCKER_VER))\n"
	@printf "$(padded_str)DOCKER_FLAG, additional docker build flags (default: $(DOCKER_FLAG))\n"
	@printf "\n"
	@printf "======= Test and linting =======\n"
	@printf "$(pretty_command): run all test and linting automations using tox\n" test
	@printf "$(pretty_command): run all style checking and linting automation using tox \n" lint
	@printf "$(pretty_command): run flake8 for style guide (PEP8) checking using tox\n" flake8
	@printf "$(pretty_command): run pylint using tox\n" pylint
	@printf "$(pretty_command): sorts python imports\n" isort

default: purge $(VENV_NAME) install

dev: purge $(VENV_NAME) install-dev-requirements

all: purge $(VENV_NAME) install-all

purge: clean-pyc clean-build clean-test clean-$(VENV_NAME)

clean: clean-pyc clean-build clean-test

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-build: ## remove build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	rm -rf .pytest_cache
	find . -name '.eggs' -type d -exec rm -rf {} +
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -f {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

clean-$(VENV_NAME):
	rm -rf $(VENV_NAME)

$(VENV_NAME):
	python3 -m virtualenv -p $(VENV_INTERP) $(VENV_NAME)

install: $(VENV_NAME)
	source $(VENV_NAME)/bin/activate ; \
	pip install --upgrade pip ; \
	if [ "$(PIP_INST_EXTRA)" = "" ]; then \
        python -m pip $(PIP_FLAG) install $(PIP_INST_FLAG) .; \
	else \
	    python -m pip $(PIP_FLAG) install $(PIP_INST_FLAG) .[$(PIP_INST_EXTRA)]; \
    fi

install-all: PIP_INST_FLAG:=$(PIP_INST_EDIT)
install-all: PIP_INST_EXTRA:=$(PIP_INST_ALL)
install-all: install

install-requirements: $(VENV_NAME)
	source $(VENV_NAME)/bin/activate ; \
	pip install -r $(REQUIREMENTS_FILE)

install-dev-requirements: $(VENV_NAME)
	source $(VENV_NAME)/bin/activate ; \
	pip install -r $(DEV_REQUIREMENTS_FILE)

docker-build:
	docker build . \
		-f $(DOCKER_FILE) \
		-t $(DOCKER_TAG):$(DOCKER_VER) \
		$(DOCKER_FLAG)

tox:
	source $(VENV_NAME)/bin/activate ; \
	tox

lint:
	source $(VENV_NAME)/bin/activate ; \
	tox -e lint

flake8:
	source $(VENV_NAME)/bin/activate ; \
	tox -e flake8

pylint:
	source $(VENV_NAME)/bin/activate ; \
	tox -e pylint

isort:
	source $(VENV_NAME)/bin/activate ; \
	isort --skip-glob=.tox --recursive . 
