"""
Define the base classes for Pii Tasks
"""

import regex
from typing import Iterable, Callable

from ..piientity import PiiEntity
from .normalizer import normalize
from .context import context_spec, context_check, CONTEXT_NORM_OPTIONS
from .exception import PiiUnimplemented


NORM_OPTIONS = dict(whitespace=True, lowercase=True)


class BasePiiTask:
    """
    Base class for a Pii Task
    """

    def __init__(self, **kwargs):
        """
        Base constructor: fetch & store all generic parameters
        """
        # print("INIT", kwargs)
        # Full set
        self.options = kwargs.copy()
        # Compulsory fields
        for f in ("pii", "lang"):
            setattr(self, f, kwargs.pop(f))
        # Optional fields
        for f in ("country", "name"):
            setattr(self, f, kwargs.pop(f, None))
        # Documentation
        self.doc = kwargs.pop("doc", self.pii.name)
        # Context
        context = kwargs.pop("context", None)
        self.context = context_spec(context) if context else None

    def context_check(self, doc: str, pii: PiiEntity) -> bool:
        """
        Check that a pii has the proper context around it
        """
        return context_check(doc, self.context, [pii.pos, pii.pos + len(pii)])

    def find_context(self, doc: str) -> Iterable[PiiEntity]:
        """
        Wrap over the standard find() method and filter out the occcurences
        that do not match a context around them
        """
        ndoc = None
        for pii in self.find(doc):
            if ndoc is None:
                ndoc = normalize(doc, self.lang, **CONTEXT_NORM_OPTIONS)
            if self.context_check(ndoc, pii):
                yield pii

    def find(self, doc: str) -> Iterable[PiiEntity]:
        """
        **Method to be implemented in subclasses**
        """
        raise PiiUnimplemented("missing implementation for Pii Task")

    def __call__(self, doc: str) -> Iterable[PiiEntity]:
        """
        Perform Pii extraction
        """
        return self.find_context(doc) if self.context else self.find(doc)

    def __repr__(self) -> str:
        """
        Return a string with a representation for the task
        """
        return f"<{self.pii.name}:{self.name}:{self.country}:{self.__class__.__qualname__}>"


class RegexPiiTask(BasePiiTask):
    """
    A wrapper for a PII implemented as a regex pattern
    Instead of the standard re package it uses the regex package (in
    backwards-compatible mode)
    """

    def __init__(self, pattern: str, **kwargs):
        super().__init__(**kwargs)
        self.regex = regex.compile(pattern, flags=regex.X | regex.VERSION0)

    def find(self, doc: str) -> Iterable[PiiEntity]:
        """
        Iterate over the regex and produce Pii objects
        """
        for cc in self.regex.finditer(doc):
            yield PiiEntity(
                self.pii, cc.start(), cc.group(), name=self.name, country=self.country
            )


class CallablePiiTask(BasePiiTask):
    """
    A wrapper for a PII implemented as a function
    """

    def __init__(self, call: Callable, extra_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.call = call
        self.kwargs = extra_kwargs or {}

    def find(self, doc: str) -> Iterable[PiiEntity]:
        """
        Call the function, get all returned strings, and locate them in the
        passed document to generate the Pii objects
        """
        for cc in self.call(doc, **self.kwargs):
            start = 0
            while True:
                pos = doc.find(cc, start)
                if pos < 0:
                    break
                yield PiiEntity(self.pii, pos, cc, name=self.name, country=self.country)
                start = pos + len(cc)
