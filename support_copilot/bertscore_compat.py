"""BERTScore 0.3.13 adapter for Transformers 5's removed empty-input helper."""


def install_empty_input_compatibility():
    from bert_score import utils
    if getattr(utils.sent_encode,'_support_compat',False):
        return
    original=utils.sent_encode

    def encode(tokenizer,sentence):
        if not sentence.strip() and not hasattr(tokenizer,'build_inputs_with_special_tokens'):
            # Equivalent empty sequence for the DistilBERT/BERT tokenizer used
            # here. BERTScore detects the two-special-token input as empty and
            # assigns zero scores, rather than inventing a placeholder reply.
            if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
                raise ValueError('Unsupported tokenizer for the empty-input adapter.')
            return [tokenizer.cls_token_id,tokenizer.sep_token_id]
        return original(tokenizer,sentence)

    encode._support_compat=True
    utils.sent_encode=encode
