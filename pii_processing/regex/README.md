
A rulebase is an ordered list of rule groups and the number of times to apply the rule groups.
A rule group is on oredered list of rules of the form (new_label, regex, old_label, before text, after text)
A rule will match if all of regex, old_label, before text and after text matches. 
 - new_label is the label to tag the matching text
 - regex is the regex to use
 - old label is the label that this text was previously tagged as.  None means to ignore the test.  
 - before text is some text before the matching pattern. None means to ignore the test.
 - after text is some text after the matching pattern. None means to ignore the test.
