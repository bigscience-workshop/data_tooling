#  Manage package tasks
#  -----------------------------------
#  make pkg       -> build the package
#  make unit      -> perform unit tests
#  make install   -> install the package in a virtualenv
#  make uninstall -> uninstall the package from the virtualenv

# Package name
NAME := pii-manager

# Virtualenv to install in. In this order:
#   1. the one given by the VENV environment variable
#   2. an active one (as given by the VIRTUAL_ENV environment variable)
#   3. a default
VENV ?= $(shell echo $${VIRTUAL_ENV:-/opt/venv/bigscience})

PYTHON ?= python3

# --------------------------------------------------------------------------

# Package version: taken from the __init__.py file
VERSION_FILE := src/pii_manager/__init__.py
VERSION	     := $(shell grep VERSION $(VERSION_FILE) | sed -r "s/VERSION = \"(.*)\"/\1/")

PKGFILE := dist/$(NAME)-$(VERSION).tar.gz

# --------------------------------------------------------------------------

all:

build pkg: $(PKGFILE)

clean:
	rm -f "$(PKGFILE)"

rebuild: clean build

version:
	@echo "$(VERSION)"

# --------------------------------------------------------------------------

TEST ?= test/unit


venv: $(VENV)

pytest: $(VENV)/bin/pytest

unit: venv pytest
	PYTHONPATH=src:test $(VENV)/bin/pytest $(ARGS) $(TEST)

unit-verbose: venv pytest
	PYTHONPATH=src:test $(VENV)/bin/pytest -vv --capture=no $(ARGS) $(TEST)

# --------------------------------------------------------------------------

$(PKGFILE): $(VERSION_FILE) setup.py
	$(PYTHON) setup.py sdist

install: $(PKGFILE)
	$(VENV)/bin/pip install $(PKGFILE)

uninstall:
	$(VENV)/bin/pip uninstall -y $(NAME)

reinstall: uninstall clean pkg install


$(VENV):
	BASE=$$(basename "$@"); test -d "$$BASE" || mkdir -p "$$BASE"
	$(PYTHON) -m venv $@
	$@/bin/pip install -r requirements.txt

$(VENV)/bin/pytest:
	$(VENV)/bin/pip install pytest


# -----------------------------------------------------------------------

upload-check: $(PKGFILE)
	twine check $(PKGFILE)

upload-test: $(PKGFILE)
	twine upload --repository pypitest $(PKGFILE)

upload: $(PKGFILE)
	twine upload $(PKGFILE)
