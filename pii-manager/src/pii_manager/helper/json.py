"""
Provide a custom JSON encoder that can serialize additional objects,
in particular PiiEntity objects
"""


from collections.abc import Iterator
import datetime
import json


def keygetter_set(v):
    return str(v).lower()


class CustomJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that can serialize additional objects:
      - datetime objects (into ISO 8601 strings)
      - sets (as sorted lists)
      - iterators (as lists)
      - any object having a to_json() method that produces a string or
        a serializable object

    Non-serializable objects are converted to plain strings.
    """

    def default(self, obj):
        """
        Serialize some special types
        """
        if hasattr(obj, "to_json"):
            return obj.to_json()
        elif isinstance(obj, datetime.datetime):
            t = obj.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
            if obj.tzinfo is not None:
                t = t[:-2] + ":" + t[-2:]
            return t
        elif isinstance(obj, set):
            return sorted(obj, key=keygetter_set)
        elif isinstance(obj, Iterator):
            return list(obj)

        try:
            return super().default(self, obj)
        except TypeError:
            return str(obj)
