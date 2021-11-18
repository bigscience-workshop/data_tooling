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
    * The task descriptor, a list with the (compulsory) name `PII_TASKS` (see
      below)
 5. Finally, add a unit test to check the validity for the task code, in the
    proper place under [test/unit/lang]. There should be at least a positive
    and a negative test

## Task implementation

A task can be implemented with either of three shapes: regex, function or
class. See [tasks] for a description of how to implement each of these,
including where to add the required documentation explaining the task.


## Task descriptor

The task descriptor is a Python list that contains at least one tuple defining
the entry points for this task (there might be more than one, if the file
implements more than one PII).

* The name of the list **must be** `PII_TASKS`

* Each defined task must be a 2- or 3-element tuple, with these elements:
   - the PII identifier for the task: a member of [PiiEnum]
   - the [task implementation]: the regex, function or class implementing the
     PII extractor
   - (only if the implementation is of regex type) a text description of the 
     task (for documentation purposes)



[task implementation]: #task-implementation
[PiiEnum]: ../src/pii_manager/piienum.py
[tasks]: tasks.md
[lang]: ../src/pii_manager/lang
[test/unit/lang]: ../test/unit/lang
[ISO 639-1]: https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
[ISO 3166-1]: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
