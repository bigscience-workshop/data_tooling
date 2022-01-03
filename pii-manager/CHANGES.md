v. 0.5.0
 * new task list parsing code, adding a "full" format based on dicts, in
   addition to the previous "simplified" format based on tuples
 * refactored to allow more than one task for a given PII and country
 * added the capability to add task descriptors programmatically
 * added reading task descriptors from a JSON file
 * context validation spec, for all three task implementation types
 * TASK_ANY split into LANG_ANY & COUNTRY_ANY
 * PII detectors for international phone numbers, for en-any & es-any
 * PII detector for IP addresses, language independent
 * PII detectors for GOV_ID
    - lang pt, countries PT & BR
    - lang es, country MX

v. 0.4.0
 * PII GOV_ID task for es-ES and en-AU
 * PII EMAIL_ADDRESS task
 * PyPi Makefile targets; fixed setup.py

v. 0.3.0
 * new processing mode: `full`
 * PII detectors for zh-CN
 * added `regex` as dependency
 * `regex` used for regular expression tasks instead of `re`

v. 0.2.0
 * Added PII tasks:
    - en: GOV_ID for US, CA, IN
    - fr: GOV_ID for CA
 * fix paths for languages/countries that are reserved Python words (is, in)
 * added country information to PiiEntity
 * added an _asdict() function for PiiEntities
 * added PII country to task_info
 * miscellaneous fixes
