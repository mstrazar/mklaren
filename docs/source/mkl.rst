Multiple kernel learning methods
--------------------------------


Mklaren - Simultaneous multiple kernel learning and low-rank approximation
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. automodule:: mklaren.mkl.mklaren

    .. autoclass:: Mklaren

        .. automethod:: __init__
        
        .. automethod:: __call__

        .. automethod:: __getitem__
        
        .. automethod:: fit
        
        .. automethod:: predict


Uniform - trivial combination of kernels
+++++++++++++++++++++++++++++++++++++++
.. automodule:: mklaren.mkl.uniform

    .. autoclass:: UniformAlignment
        :members:

        .. automethod:: __call__

        .. automethod:: __getitem__
    
    .. autoclass:: UniformAlignmentLowRank
        :members:

        .. automethod:: __call__

        .. automethod:: __getitem__


Align - Independent centered alignment
++++++++++++++++++++++++++++++++++++++
.. automodule:: mklaren.mkl.align

    .. autoclass:: Align
        :members:

        .. automethod:: __call__

        .. automethod:: __getitem__
    
    .. autoclass:: AlignLowRank
        :members:

        .. automethod:: __call__

        .. automethod:: __getitem__


Alignf - Optimizing the linear combination of kernels
+++++++++++++++++++++++++++++++++++++++++++++++++++++
.. automodule:: mklaren.mkl.alignf

    .. autoclass:: Alignf
        :members:

        .. automethod:: __init__

        .. automethod:: __call__

        .. automethod:: __getitem__
    
    .. autoclass:: AlignfLowRank
        :members:
        
        .. automethod:: __init__

        .. automethod:: __call__

        .. automethod:: __getitem__

