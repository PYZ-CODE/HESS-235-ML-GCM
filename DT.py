#Apply Decision Tree to produce ensemble dataset and calculate the accuray 

from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import math
from sklearn import metrics
import BaseFunc

class DecisionTreeRegressor(RegressorMixin, BaseDecisionTree):
    """A decision tree regressor.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"squared_error", "friedman_mse", "absolute_error", \
            "poisson"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

        .. versionadded:: 0.24
            Poisson deviance criterion.

        .. deprecated:: 1.0
            Criterion "mse" was deprecated in v1.0 and will be removed in
            version 1.2. Use `criterion="squared_error"` which is equivalent.

        .. deprecated:: 1.0
            Criterion "mae" was deprecated in v1.0 and will be removed in
            version 1.2. Use `criterion="absolute_error"` which is equivalent.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, de 
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

        .. deprecated:: 1.0
           `n_features_` is deprecated in 1.0 and will be removed in
           1.2. Use `n_features_in_` instead.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    DecisionTreeClassifier : A decision tree classifier.


    
    """

    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )

    def fit(
        self, X, y, sample_weight=None, check_input=True, X_idx_sorted="deprecated"
    ):
        """Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers). Use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        X_idx_sorted : deprecated, default="deprecated"
            This parameter is deprecated and has no effect.
            It will be removed in 1.1 (renaming of 0.26).

            .. deprecated:: 0.24

        Returns
        -------
        self : DecisionTreeRegressor
            Fitted estimator.
        """

        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted,
        )
        return self

    def _compute_partial_dependence_recursion(self, grid, target_features):
        """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray of shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray of shape (n_target_features)
            The set of target features for which the partial dependence
            should be evaluated.

        Returns
        -------
        averaged_predictions : ndarray of shape (n_samples,)
            The value of the partial dependence function on each grid point.
        """
        grid = np.asarray(grid, dtype=DTYPE, order="C")
        averaged_predictions = np.zeros(
            shape=grid.shape[0], dtype=np.float64, order="C"
        )

        self.tree_.compute_partial_dependence(
            grid, target_features, averaged_predictions
        )
        return averaged_predictions

    @deprecated(  # type: ignore
        "The attribute `n_features_` is deprecated in 1.0 and will be removed "
        "in 1.2. Use `n_features_in_` instead."
    )
    @property
    def n_features_(self):
        return self.n_features_in_



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
            
        #construct DT model clf
        clf = tree.DecisionTreeRegressor()
        clf.fit(x_train, y_train)

        #Apply reg to predict y for validation dataset  which represent the ensemble dataset. Moreover assessment the accuracy of DT
        y_test_pre = clf.predict(x_test)
        y_test_pre=np.reshape(y_test_pre,())

        #Calculate R
        R=np.corrcoef(np.array(y_test), np.array(y_pre_test))[0][1]
      
        #Produce dataset
        for year in range(2015,2101,1):
                    ssp_path_predict=r'F:\6_PreSSP'+"\\"+var+"\\"+ssp+"\\"+mon+"\\"+str(year)
                    x_predict=[] ; y_predict=[]
                    y_ssp_ori=BaseFunc.GetSSPDataset(ssp_path_predict,true_pixel_index)
                    y_ssp_pre=clf.predict(y_ssp_ori)
                    ssp_tif_folder=r'F:\6_PreSSP\TIF\OLS'+"\\"+var+"\\"+ssp+"\\"+str(year)
                    if not os.path.isdir(ssp_tif_folder):
                        ssp_tif_path=ssp_tif_folder+"\\"+"OLS_"+var+"_"+ssp+"_"+str(year)+"_"+str(mon_num)+".tif"
                    BaseFunc.WriteSSPTIF(true_pixel_index,y_ssp_pre,ssp_tif_path)
      
       
        


        
