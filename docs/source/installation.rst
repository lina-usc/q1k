Installation
============

Developer installation
----------------------

Clone the repository and install the package in editable mode:

.. code-block:: bash

   git clone https://github.com/lina-usc/q1k.git
   cd q1k
   python -m pip install --upgrade pip
   python -m pip install -e ".[test,doc,dev]"

Runtime installation
--------------------

For a processing environment that does not need documentation or release tools:

.. code-block:: bash

   python -m pip install -e .

Optional analysis dependencies
------------------------------

Group-level notebooks and downstream statistical analyses require extra
packages:

.. code-block:: bash

   python -m pip install -e ".[analyses]"
