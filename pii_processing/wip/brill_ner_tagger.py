  
def learn_brill_transformation_base_tagger(dat, regexs, target_lang, pattern_search_window = 3, ntimes=4):
  """ 
  Input:
    dat: the corpus plus init ner tagging in list of dict. Assumes a basic lexcion or other process performed an init NER tag on dat.
    regexes: Regexs used to detect and re-tag to reduce the error. 
    target_lang: used for shingling, conversion, etc.
    pattern_search_window: number of words or tags before and after regex matched pattern used to create rules
    ntimes: the maximum number of iterations through the alogirthm.

  Output:
    rules: ordered list of (regex, matching nearby shingles and tag patterns, tag_wrong => tag_right, score)
  
  Side effect: dat['predicted_ner'] will be changed to the prediction applied by the returned rules

  Naive algortihm (very memory inefficient)
  TODO for participants is to find a faster algortihm, including out of core or dynamic programming (caching ngram).
  
  loop ntimes or there are no more improvements 
    given a set of sentences with predict_ner and actual ner, 
      - find which regex that will match an item that is wrongly tagged by predict_ner, 
      - find nearby word shingles
      - find nearby tag patterns
      - add candidate (regex, ngram pattern before, ngram pattern after, tag_wrong => tag_right, number of wrong matches, number of right matches)  
    find the top n rules that do not have overlapping patterns or shingles, and maximizes the score with number of right > X
    apply the top n rules to create a new predict_ner, and continue loop

  """
  ontology_manager = OntologyManager(target_lang=target_lang)
  tag2regex = {}
  regex_id = 0
  for tag, regex in regexes:
    tag2regex[tag] = tag2regex.get(tag, []) + [(regex_id, regex)]
    regex_id += 1

  for times in range(ntimes):
    candidates = {}
    for d in dat:
      text = d['text']
      ner = d['ner'] # the human tagged data
      predicted_ner = d['predicted_ner']

      #find tagging that are wrong
      for ent, tag in ner.items(): # sort by sentence order
        if tag not in tag2regex: continue
        if ent not in predicted_ner  predicted_ner[ent] != tag:
          regex_id, regex = tag2regex[tag]
          match = regex.findall(ent)
          if not match: continue
          before, after = find_before_after_words(text, ent, pattern_search_window)
          before_shingles = ontology_manager._get_shingles(before)
          after_shingles = ontology_manager._get_shingles(after)
          # before, after correct tags
          for before, after in all_combos(before_shingles, after_shingles):
            candidates[(before, after, predicted_ner.get(ent), tag, regex_id)] = 0

      # find tagging that shouldn't be done at all
      for ent, tag in predicted_ner.items():
        # do a case where there are no regexes
        if tag not in tag2regex: continue
        if ent not in ner:
          regex_id, regex = tag2regex[tag]
          match = regex.findall(ent)
          if not match: continue
          before, after = find_before_after_words(text, ent, pattern_search_window)
          before_shingles = ontology_manager._get_shingles(before)
          after_shingles = ontology_manager._get_shingles(after)
          # before, after correct tags
          for before, after in all_combos(before_shingles, after_shingles):
            candidates[(before, after, tag, None, regex_id)] = 0

