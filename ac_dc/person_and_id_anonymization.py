from muliwai.regex_manager import detect_ner_with_regex_and_context
from muliwai.pii_regexes_rulebase import regex_rulebase
from muliwai.ner_manager import detect_ner_with_hf_model
from muliwai.faker_manager import augment_anonymize

def apply_anonymization(
    sentence: str,
    lang_id: str,
    context_window: int = 20,
    anonymize_condition=None,
    tag_type={'IP_ADDRESS', 'KEY', 'ID', 'PHONE', 'USER', 'EMAIL', 'LICENSE_PLATE', 'PERSON'} ,
    device: str = "cpu",
) -> str:
    """
    Params:
    ==================
    sentence: str, the sentence to be anonymized
    lang_id: str, the language id of the sentence
    context_window: int, the context window size
    anonymize_condition: function, the anonymization condition
    tag_type: iterable, the tag types of the anonymization. By default: {'IP_ADDRESS', 'KEY', 'ID', 'PHONE', 'USER', 'EMAIL', 'LICENSE_PLATE', 'PERSON'} 
    device: cpu or cuda:{device_id}

    """
    if tag_type == None:
        tag_type = regex_rulebase.keys()
    lang_id = lang_id.split("_")[0]
    ner_ids = detect_ner_with_regex_and_context(
        sentence=sentence,
        src_lang=lang_id,
        context_window=context_window,
        tag_type=tag_type,
    )
    ner_persons = detect_ner_with_hf_model(
        sentence=sentence,
        src_lang=lang_id,
        device=device,
    )
    ner = ner_ids + ner_persons
    if anonymize_condition:
        new_sentence, new_ner, _ = augment_anonymize(sentence, lang_id, ner, )
        doc = {'text': new_sentence, 'ner': new_ner, 'orig_text': sentence, 'orig_ner': ner}
    else:
        new_sentence = sentence
        doc = {'text': new_sentence, 'ner': ner}
    return new_sentence, doc
