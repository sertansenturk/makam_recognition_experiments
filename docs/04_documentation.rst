
*****************************************
Documentation
*****************************************

The project comes with basic documentation, which is located at ``makam_recognition_experiments/docs``. You can use `Sphinx <https://www.sphinx-doc.org>`__ to build the documentation locally by running:

.. code:: bash

   make sphinx-html

The above command builds a docker image called ``sertansenturk/makam_recognition_experiments/sphinx``. It then starts a container from the image and renders the documentation (including automatic Python API documentation from docstrings).

Afterward, you can access the documentation by opening ``./docs/_build/html/index.html`` on your browser.
