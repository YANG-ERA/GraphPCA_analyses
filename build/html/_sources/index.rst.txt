.. GraphPCA documentation master file, created by
   sphinx-quickstart on Wed Apr  3 19:25:25 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GraphPCA's documentation!
====================================

GraphPCA – a fast and interpretable dimension reduction algorithm for spatial transcriptomics data.
=====================================================================================================


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Installation
   Tutorial1_DLPFC
   Tutorial2_mPFC
   Tutorial3_Integration

.. image:: ../figures/workflow.png
   :width: 1400

Overview
========
GraphPCA is a novel graph-constrained, interpretable,
and quasi-linear dimension-reduction method tailored for spatial transcriptomic data.
It leverages the strengths of graphical regularization and Principal Component Analysis (PCA)
to extract low-dimensional embeddings of spatial transcriptomes that integrate location information in linear time complexity.
The substantial power boost enabled by GraphPCA fertilizes various downstream tasks of spatial transcriptomics data analyses and
provides more precise insights into transcriptomic and cellular landscapes of complex tissues.
