# Adding external task processors to a processing object

In addition to the task processorts contained inside the [lang] subfolders in
the package, it is also possible to add _external_ task processors define
outside the package, as long as they comply with the [task specification].
This can be done for both the object-base API and the file-based API.


## Object-based API

An instantiated `ProcManager` object contains the `add_tasks` method. This
method will accept a list of [task descriptors] with the same syntax as the
internal `PII_TASKS` descriptors, and will add the tasks defined in them to
the existing ones in the object.


## File-based API

The file-based `process_file` function allows a `taskfile` argument. This
argument will contain the name of a JSON file that contains an array of task
descriptors. Each task descriptor in the array is a JSON object following the
specification for [task descriptors], with these differences:

* The `pii` field is not a `PiiEnum` object, but a string with the _name_ of
  a `PiiEnum` object. It will be converted to the object itself.
* The `task` field contains:
   - for `regex` types, the string with the regular expression pattern to be
     compiled (beware of escaping all backlashes in the string)
   - for `callable` and `PiiTask` types, a string with the **fully
     qualified** name of the function to be used or class to be instantiated.
     As long as that name can be located in the running Python space (i.e.
     it is in the load path), it will be imported and used.


[lang]: ../src/pii_manager/lang
[task specification]: tasks.md
[task descriptors]: contributing.md#task-descriptor
