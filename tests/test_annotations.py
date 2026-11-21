import pytest
from support_copilot.annotations import validate_labels,compare_raters

SOURCE=[{'case_id':'c1','brand':'Delta','customer_text':'My bag did not arrive.'}]


def test_import_keeps_source_text_and_rejects_duplicate_ids():
    rows=[{'case_id':'c1','brand':'Wrong','customer_text':'Rewritten','intent':'delay','urgency':'med','needs_human':False}]
    clean=validate_labels(rows,SOURCE)
    assert clean[0]['brand']=='Delta' and clean[0]['customer_text']==SOURCE[0]['customer_text']
    assert clean[0]['needs_human']=='false'
    with pytest.raises(ValueError):
        validate_labels(rows+rows,SOURCE)


def test_invalid_label_is_rejected():
    with pytest.raises(ValueError):
        validate_labels([{'case_id':'c1','intent':'invented'}],SOURCE)


def test_empty_annotations_do_not_become_zero_accuracy():
    rows=validate_labels([{'case_id':'c1'}],SOURCE)
    report=compare_raters(rows,rows)
    assert report['metrics']['intent']['agreement'] is None
    assert report['metrics']['intent']['paired_labels']==0
