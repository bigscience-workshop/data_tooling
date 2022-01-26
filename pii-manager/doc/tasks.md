# PII Task Implementation

A PII task is a module that provides detection of a given PII (possible for a
given language and country). The `pii-manager` package accepts three types of
implementations for a PII task. They are commented in the next sections.


## Regex implementation

In its simplest form, a PII Task can be just a regular expression
pattern. This pattern should match string fragments that correspond to the PII
entity to be detected.

Rules for the implementation of such regex are:

* Implement it as a regular expression _string_, **not** as a compiled regular
  expression (it will be compiled by the `pii-manager` module)
* The pattern **will be compiled with the [regex] package**, instead of the
  `re` package in the standard Python library, so you can use the extended
  features (such as unicode categories) in `regex`. Compilation will be done
  in backwards-compatible mode (i.e. using the `VERSION0` flag), so it should
  be fully compatible with `re`
* Do **not** anchor the regex to either beginning or end (it should be able to
  match anywhere in the passed string, which itself can be any portion of
  a document)
* Do not include capturing groups (they will be ignored)
* The pattern will be compiled with the [re.VERBOSE] (aka `re.X`) flag, so
  take that into account (in particular, **whitespace is ignored**, so if it is
  part of the regular expression needs to included as a category i.e. `\s`, or
  escaped)

An example can be seen in the [US Social Security Number] detector.


## Callable implementation

The next implementation type is via a function. The signature for the function
is:

```Python

   def my_pii_detector(src: doc) -> Iterable[str]:
```

The function can have any name, but it should indicate the entity it is
capturing, since it will be used as the `name` attribute for the task (after
converting underscores into spaces).

The function should:

 * accept a string: the document to analyze
 * return an iterable of strings: the string fragments corresponding to the
   PII entities identified in the document

An example can be seen in the [bitcoin address] detector.

**Note**: in case the same entity appears more than once in the passed
document, it might be possible that the callable returns repeated strings.
This is not a problem; all of them will be reported.

Conversely, if a given string in the document is a PII some of the time but
it also appears in a non-PII role in the same document, the wrapper that uses
the result of a callable implementation type will not be able to differentiate
among them, and the package will label *all* ocurrences of the string as PII.
If this is likely to happen, and there is code that *can* separate both uses,
then it is better to use the class implementation type below.


## Class implementation

In this case the task is implemented as a full Python class. The class *must*:

 * inherit from `pii_manager.helper.BasePiiTask`
 * implement a `find` method with the following signature:

        def find(self, doc: str) -> Iterable[PiiEntity]:

   i.e. a method returting an iterable of identified [PiiEntity]

 * the default task name will be taken from the class-level attribute
   `pii_name`, if it exists, or else as the class name. Nevertheless, the name
   can be dynammicaly set with each detected PiiEntity

The class can also, optionally, include a constructor. In this case, the
constructor must
 * accept an arbitrary number of keyword arguments
 * call the parent class constructor with those arguments

In other words:

```Python

   def __init__(self, **kwargs):
     super().__init__(**kwargs)
     ... any code specific to the implementation here ...
```


Examples can be seen in the [credit card] detector or the [Spanish GOV ID]
detector.


## Documentation

In addition to its name, all PII Tasks should be documented with a small
string that explains what they detect. The place to add this documentation is:
 * For Regex tasks, add the string as the third element in the task descriptor
   inside `PII_TASKS`
 * For Callable tasks, use the function docstring to add the documentation.
 * For Class tasks, add the documentation as the _class level_ docstring.

For any task type, if using the "full" description in `PII_TASKS`, a `doc`
field can be added to the dictionary description, and it will override any
automatic generation from docstrings.


[regex]: https://github.com/mrabarnett/mrab-regex
[US Social Security Number]: ../src/pii_manager/lang/en/us/social_security_number.py
[bitcoin address]: ../src/pii_manager/lang/any/bitcoin_address.py
[credit card]: ../src/pii_manager/lang/any/credit_card.py
[Spanish GOV ID]: ../src/pii_manager/lang/es/es/govid.py
[PiiEntity]: ../src/pii_manager/piientity.py
[re.VERBOSE]: https://docs.python.org/3/library/re.html#re.X
