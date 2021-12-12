v. 0.5.0
 * new task list parsing code, adding a "full" format based on dicts, in
   addition to the previous "simplified" format based on tuples

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
