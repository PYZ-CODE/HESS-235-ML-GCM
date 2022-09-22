#Apply OLS to produce ensemble dataset and calculate the accuray 

import os
from osgeo import gdal,osr,ogr
from sklearn import linear_model
import numpy as np
import glob
from pathlib import Path
import time
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
import BaseFunc
class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
    """
    Ordinary least squares Linear Regression.

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

        .. deprecated:: 1.0
           `normalize` was deprecated in version 1.0 and will be
           removed in 1.2.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This will only provide
        speedup in case of sufficiently large problems, that is if firstly
        `n_targets > 1` and secondly `X` is sparse or if `positive` is set
        to `True`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive. This
        option is only supported for dense arrays.

        .. versionadded:: 0.24

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    rank_ : int
        Rank of matrix `X`. Only available when `X` is dense.

    singular_ : array of shape (min(X, y),)
        Singular values of `X`. Only available when `X` is dense.

    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if
        `fit_intercept = False`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    Ridge : Ridge regression addresses some of the
        problems of Ordinary Least Squares by imposing a penalty on the
        size of the coefficients with l2 regularization.
    Lasso : The Lasso is a linear model that estimates
        sparse coefficients with l1 regularization.
    ElasticNet : Elastic-Net is a linear regression
        model trained with both l1 and l2 -norm regularization of the
        coefficients.

    Notes
    -----
    From the implementation point of view, this is just plain Ordinary
    Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares
    (scipy.optimize.nnls) wrapped as a predictor object.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.score(X, y)
    1.0
    >>> reg.coef_
    array([1., 2.])
    >>> reg.intercept_
    3.0...
    >>> reg.predict(np.array([[3, 5]]))
    array([16.])
    """

    def __init__(
        self,
        *,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.

        Returns
        -------
        self : object
            Fitted Estimator.
        """

        _normalize = _deprecate_normalize(
            self.normalize, default=False, estimator_name=self.__class__.__name__
        )

        n_jobs_ = self.n_jobs

        accept_sparse = False if self.positive else ["csr", "csc", "coo"]

        X, y = self._validate_data(
            X, y, accept_sparse=accept_sparse, y_numeric=True, multi_output=True
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            normalize=_normalize,
            copy=self.copy_X,
            sample_weight=sample_weight,
            return_mean=True,
        )

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        if self.positive:
            if y.ndim < 2:
                self.coef_, self._residues = optimize.nnls(X, y)
            else:
                # scipy.optimize.nnls cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(optimize.nnls)(X, y[:, j]) for j in range(y.shape[1])
                )
                self.coef_, self._residues = map(np.vstack, zip(*outs))
        elif sp.issparse(X):
            X_offset_scale = X_offset / X_scale

            def matvec(b):
                return X.dot(b) - b.dot(X_offset_scale)

            def rmatvec(b):
                return X.T.dot(b) - X_offset_scale * np.sum(b)

            X_centered = sparse.linalg.LinearOperator(
                shape=X.shape, matvec=matvec, rmatvec=rmatvec
            )

            if y.ndim < 2:
                out = sparse_lsqr(X_centered, y)
                self.coef_ = out[0]
                self._residues = out[3]
            else:
                # sparse_lstsq cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(sparse_lsqr)(X_centered, y[:, j].ravel())
                    for j in range(y.shape[1])
                )
                self.coef_ = np.vstack([out[0] for out in outs])
                self._residues = np.vstack([out[3] for out in outs])
        else:
            self.coef_, self._residues, self.rank_, self.singular_ = linalg.lstsq(X, y)
            self.coef_ = self.coef_.T

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self


mon_inEng={'Apr'}#Just  a example for training OLS model for April
var='pr'
ssp='ssp126'#or ssp 245 or ssp 585
if __name__ == '__main__':
    

    obs_img_path=r'F:\TestDemo1\obs_pr__1967_2.tif'
    true_pixel_index = BaseFunc.GetTPixelIndex(obs_img_path)
    R_accuracy_list=list(range(12))
    R2_accuracy_list=list(range(12))
    std_accuracy_list=list(range(12))
    CRMSE_accuracy_list=list(range(12))
    MAE_accuracy_list=list(range(12))
    for mon in mon_inEng:
        mon_num=BaseFunc.GetMonthNum(mon)
        month_path_train=r'F:\5_TrainingDataSet\pr'+"\\"+mon+"\\TrainNet"
        year_folders_list_train=BaseFunc.GetSubfoldOfMonth(month_path_train)
        month_path_test=r'F:\5_TrainingDataSet\pr'+"\\"+mon+"\\TestNet"
        year_folders_list_test=BaseFunc.GetSubfoldOfMonth(month_path_test)
        
    
        x_train=[] ; y_train=[]
        BaseFunc.GetXYDataset(year_folders_list_train,x_train,y_train,true_pixel_index)
        x_test=[] ; y_test=[]
        BaseFunc.GetXYDataset(year_folders_list_test,x_test,y_test,true_pixel_index)
            
        #construct OLS model reg
        reg = linear_model.LinearRegression()
        reg.fit(x_train, y_train)
        result = reg.coef_

        #Apply reg to predict y for validation dataset  which represent the ensemble dataset. Moreover assessment the accuracy of OLS
        y_test_pre = reg.predict(x_test)
        y_test_pre=np.reshape(y_test_pre,())

        #Calculate R
        R=np.corrcoef(np.array(y_test), np.array(y_pre_test))[0][1]
      
        #Produce dataset
        for year in range(2015,2101,1):
                    ssp_path_predict=r'F:\6_PreSSP'+"\\"+var+"\\"+ssp+"\\"+mon+"\\"+str(year)
                    x_predict=[] ; y_predict=[]
                    y_ssp_ori=BaseFunc.GetSSPDataset(ssp_path_predict,true_pixel_index)
                    y_ssp_pre=reg.predict(y_ssp_ori)
                    ssp_tif_folder=r'F:\6_PreSSP\TIF\OLS'+"\\"+var+"\\"+ssp+"\\"+str(year)
                    if not os.path.isdir(ssp_tif_folder):
                        ssp_tif_path=ssp_tif_folder+"\\"+"OLS_"+var+"_"+ssp+"_"+str(year)+"_"+str(mon_num)+".tif"
                    BaseFunc.WriteSSPTIF(true_pixel_index,y_ssp_pre,ssp_tif_path)
      
       
        


        

                  
                 


  