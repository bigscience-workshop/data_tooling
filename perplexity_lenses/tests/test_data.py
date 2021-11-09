import unittest

import pandas as pd

from perplexity_lenses.data import documents_df_to_sentences_df


class TestData(unittest.TestCase):
    def test_documents_df_to_sentences_df(self):
        input_df = pd.DataFrame({"text": ["foo\nbar"]})
        expected_output_df = pd.DataFrame({"text": ["foo", "bar"]})
        output_df = documents_df_to_sentences_df(input_df, "text", 100)
        pd.testing.assert_frame_equal(
            output_df, expected_output_df, check_like=True, check_exact=True
        )
