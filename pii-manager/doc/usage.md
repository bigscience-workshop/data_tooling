# Usage

## API Usage

There are two types of API usage: the object-based API (lower-level, based on
object instantiation) and the file-based API (higher-level, based on function
calls).


### Object API

The object-based API is centered on the `PiiManager` object. Its usage goes
like this:

```Python

 from pii_manager import PiiEnum
 from pii_manager.api import PiiManager

 # Define language, country(ies) and PII tasks
 lang = 'en'
 country = ['US', 'GB']
 tasklist = (PiiEnum.CREDIT_CARD, PiiEnum.GOVID, PiiEnum.DISEASE)

 # Instantiate object
 proc = PiiManager(lang, country, tasks=tasklist)

 # Process a text buffer
 text_out = proc(text_in)

```

... this will load and execute anonymization tasks for English that will
anonymize credit card numbers, disease information, and Government IDs for US
and UK (assuming all these tasks are implemented in the package).


It is also possible to load all possible tasks for a language, by specifying
the country as `COUNTRY_ANY` and using the `all_tasks` argument.

```Python

 from pii_manager import PiiEnum
 from pii_manager.api import PiiManager
 from pii_manager.lang import COUNTRY_ANY

 proc = PiiManager('en', COUNTRY_ANY, all_tasks=True)

 text_out = proc(text_in)

```

...this will load all anonymization tasks available for English, including:
 * language-independent tasks
 * language-dependent but country-independent tasks
 * country-dependent tasks for *all* countries implemented under the `en`
   language

Finally, the API allows for [importing arbitrary tasks] defined outside the
package.


### File-based API

The file-based API uses the `process_file` function to read from a file and
write the result to an output file. It is executed as:

```Python

 from pii_manager import PiiEnum
 from pii_manager.api import process_file

 # Define language, country(ies) and PII tasks
 lang = 'en'
 country = ['US', 'GB']
 tasklist = (PiiEnum.CREDIT_CARD, PiiEnum.GOVID, PiiEnum.DISEASE)

 # Process the file
 process_file(infilename, outfilename, lang,
              country=country, tasks=tasklist)

```

The file-based API accepts also the `all_tasks` argument to add all suitable
defined tasks, as well as the `COUNTRY_ANY` option. It can also [import
external tasks], as defined in a JSON file.


## Command-line usage

Installing the package provides also a command-line script, `pii-manage`,
that can be used to process files through PII tasks:

    pii-manage <infile> <outfile> --lang es --country es ar mx \
       --tasks CREDIT_CARD BITCOIN_ADDRESS BANK_ACCOUNT

or, to add all possible tasks for a given language:

    pii-manage <infile> <outfile> --lang es --country all \
       --all-tasks


There is an additional command-line script, `pii-task-info`, that does not
process text; it is only used to show the available tasks for a given language.


## Processing mode

PII processing accepts four modes: _replace_ , _tag_, _extract_ and _full_. To
show an example, let us consider a fragment such as:

> my credit card number is 4273 9666 4581 5642

with this input, the output for each of the processing modes will be:

* for _replace_, the PII will be replaced by a placeholder describing the PII
  name:

> my credit card number is <CREDIT_CARD>

* for _tag_, the PII is tagged with the PII name, but the original string is
  also kept:

> my credit card number is <CREDIT_CARD:4273 9666 4581 5642>

* for _extract_, a list of detected PIIs is returned, as a dict in the
  buffer-based API, or as a [NDJSON] file for the file-based API

> {"name": "CREDIT_CARD", "value": "4273 9666 4581 5642", "pos": 25, "line": 1}

* for _full_ mode, the API returns a dict or a NDJSON line for each text
  fragment, containing the fields `text` (the passed text) and `entities`
  (a list of the recognized PII entities)

> {"text": "my credit card number is 4273 9666 4581 5642",
>  "entities": [{"name": "CREDIT_CARD", "value": "4273 9666 4581 5642",
>                "pos": 25, "line": 1}]
> }


By default in _replace_ mode all PII items found will be substituted with
a `<PIINAME>` string. If another placeholder is preferred, the `PiiManager`
constructor can be called with an additional `template` argument, containing
a string that will be processed through the Python string `format()` method,
and called with a `(name=PIINAME)` argument. In _tag_ mode, the template is
called with `(country=COUNTRY, name=PIINAME, value=VALUE)` arguments,
which the template can use as it sees fit.

The file-based API has an additional option: how the file is splitted when
calling the PII tasks:

* `line`: (default) the file is splitted line-by-line
* `block`: the file is sent as a single block
* `sentences`: the file is split by sentence separators (periods, etc)


[NDJSON]: http://ndjson.org/
[importing arbitrary tasks]: external.md#object-based-api
[import external tasks]:external.md#file-based-api
