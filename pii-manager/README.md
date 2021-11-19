# Pii Manager

This repository builds a Python package that performs PII processing for text
data i.e. replacement/tagging/extraction of PII in the text.

The PII Tasks in the package are structured by language & country, since many
of the PII elements are language- and/or -country dependent.


## Usage

The package can be used:
 * As an API, in two flavors
 * As a command-line tool

For details, see [usage]


## Building

The provided [Makefile] can be used to process the package:
 * `make pkg` will build the Python package, creating a file that can be
   installed with `pip`
 * `make unit` will launch all unit tests (using [pytest], so pytest must be
   available)
 * `make install` will install the package in a Python virtualenv. The
   virtualenv will be chosen as, in this order:
     - the one defined in the `VENV` environment variable, if it is defined
     - if there is a virtualenv activated in the shell, it will be used
     - otherwise, a default is chosen (which will probably not be available)


## Contributing

To add a new PII processing task, please see the [contributing instructions]


[Makefile]: Makefile
[pytest]: https://docs.pytest.org
[contributing instructions]: doc/contributing.md
[usage]: doc/usage.md
