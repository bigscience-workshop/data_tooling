from snorkel.labeling.apply.core import BaseLFApplier, _FunctionCaller
from snorkel.labeling.apply.pandas import apply_lfs_to_data_point, rows_to_triplets
from dask.distributed import Client, Scheduler
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))

#################################################################################################################################
# Code for Snorkel label function appliers which for Datastores. 

class DatastoreLFApplier(BaseLFApplier):
    """LF applier for a Datastore DataFrame.  Datastore DataFrames
    consist of Dask dataframes in certain columns, and just-in-time
    sharded loading of data from SQL or shared drive in other columns.
    We can convert everything to pandas and use the snorkel underlying
    pandas LF.  This allows for efficient parallel computation over
    DataFrame rows.  For more information, see
    https://docs.dask.org/en/stable/dataframe.html
    """

    def apply(
        self,
        df: "Datastore",
        scheduler:  "processes",
        fault_tolerant: bool = False,
    ) -> np.ndarray:
        """Label Datastore of data points with LFs.
        Parameters
        ----------
        df
            Datastore containing data points to be labeled by LFs
        scheduler
            A Dask scheduling configuration: either a string option or
            a ``Client``. For more information, see
            https://docs.dask.org/en/stable/scheduling.html#
        fault_tolerant
            Output ``-1`` if LF execution fails?
        Returns
        -------
        np.ndarray
            Matrix of labels emitted by LFs
        """
        f_caller = _FunctionCaller(fault_tolerant)
        apply_fn = partial(apply_lfs_to_data_point, lfs=self._lfs, f_caller=f_caller)
        labels = df.map(lambda p_df: p_df.apply(apply_fn, axis=1), distributed_context=scheduler)
        labels_with_index = rows_to_triplets(labels)
        return self._numpy_from_row_data(labels_with_index)

class DatastoreSFApplier(DatastoreLFApplier):  # pragma: no cover
    """SF applier for a Datastore DataFrame.
    """

    _use_recarray = True


# TODO, all LF applications will write to a column, either mmemmap in original dataset or a new dataset.

