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
