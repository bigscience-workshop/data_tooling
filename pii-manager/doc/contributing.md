# Adding PII tasks

To add a new PII processing task to the package, prepare a Pull Request on the
repository with the following changes:

 1. If the task type is a new one, add an identifier for it in [PiiEnum]
 2. If it is for a language not yet covered, add a new language subfolder
    undder the [lang] folder, using the [ISO 639-1] code for the language
 3. Then
    * If it is a country-independent PII, it goes into the `any` subdir
      (create that directory if it is not present)
    * if it is country-dependent, create a country subdir if not present,
      using a **lowercased version** of its [ISO 3166-1] country code
 4. Under the final chosen subfolder, add the task as a Python `mytaskname.py`
    module (the name of the file is not relevant). The module must contain:
    * The task implementation, which can have any of three flavours (regex,
      function or class), see below
    * The task descriptor, a list containing all defined tasks. The list
      variable *must* be named `PII_TASKS` (see below)
 5. Finally, add a unit test to check the validity for the task code, in the
    proper place under [test/unit/lang]. There should be at least
     - a positive test: one valid PII that has to be detected. For the cases
       in which the PII is validated, the test should pass the validation,
       so the PII must be a random (fake) one but still valid
     - and a negative test: one PII-like string that is almost, but not quite
       the real one, so it should *not* be recognized


## Task implementation

A task can be implemented with either of three shapes: regex, function or
class. See [tasks] for a description of how to implement each of these,
including where to add the required documentation explaining the task.


## Task descriptor

The task descriptor is a Python list that contains at least one element 
defining the entry points for this task (there might be more than one, if 
the file implements more than one PII).

* The name of the list **must be** `PII_TASKS`
* A task entry in the list can have two different shapes: simplified and full.
  In a `PII_TASKS` list they can be combined freely.


### Simplified description

In a simplified description a task must be a 2- or 3-element tuple, with 
these elements:
   - the PII identifier for the task: a member of [PiiEnum]
   - the [task implementation]: the regex, function or class implementing the
     PII extractor
   - (only if the implementation is of regex type) a text description of the
     task (for documentation purposes)


### Full description

In a full description a task is a dictionary with these compulsory fields:
 * `pii`: the PII identifier for the task: a member of [PiiEnum]
 * `type`: the task type: `regex`, `callable` or `PiiTask`
 * `task`: for regex tasks, a raw string (contianing the regex to be used);
    for function tasks a callable and for PiiTask either a class or a string
	with a full class name.
 
And these optional fields
 * `lang`: language this task is designed for (it can also be `LANG_ANY`). If
   not present, the language will be determined from the folder structure the
   task implementation is in
 * `country`: country this task is designed for. If not present, the language
   will be determined from the folder structure the task implementation is in,
   if possible (else, a `None`value will be used, meaning the task is not
   country-dependent)
 * `name`: a name for the task. If not present, a name will be generated from
   the `name` class-level attribute (PiiTask) or from the class/function name.
   This is meant to provide a higher level of detail than the `PiiEnum`
   generic name (e.g. for different types of Government ID). Class-type tasks
   can use a dynamic name at runtime (detected PII might have different names),
   while function and regexes will have a fixed name.
 * `doc`: the documentation for the class. If not present, the docstring for
   callable and class types will be used (for regex types, the task will have
   no documentation)


[task implementation]: #task-implementation
[PiiEnum]: ../src/pii_manager/piienum.py
[tasks]: tasks.md
[lang]: ../src/pii_manager/lang
[test/unit/lang]: ../test/unit/lang
[ISO 639-1]: https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
[ISO 3166-1]: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
