################# customized jupyter image ################
FROM jupyter/scipy-notebook:862de146632b AS jupyter-custom

RUN conda install --quiet --yes -c conda-forge \
    'conda-build' \
    'tqdm' \
    'jupyter_contrib_nbextensions' \
    'jupyter_nbextensions_configurator' && \
    conda build purge-all && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

USER root
RUN ldconfig && \
    jupyter nbextension enable toc2/main --sys-prefix && \
    jupyter nbextension enable collapsible_headings/main --sys-prefix
USER $NB_UID

################# experiment image ################
FROM jupyter-custom AS experimentation

COPY --chown=$NB_UID ./src/ ./work/src/
COPY --chown=$NB_UID ./setup.py ./work/
RUN pip install ./work/

################# development image ################
FROM jupyter-custom AS development
# Install Python dependencies from requirements.txt in advance
# Useful for development since changes in code will not trigger a layer re-build
COPY --chown=$NB_UID ./.pylintrc ./work/
COPY --chown=$NB_UID ./tox.ini ./work/
COPY --chown=$NB_UID ./requirements.txt ./work/
COPY --chown=$NB_UID ./requirements.dev.txt ./work/
COPY --chown=$NB_UID ./tests/ ./work/tests/
RUN pip install --upgrade pip && \
    pip install -r ./work/requirements.txt && \
    pip install -r ./work/requirements.dev.txt

# install experimentation code in editable mode
COPY --chown=$NB_UID ./setup.py ./work/
COPY --chown=$NB_UID ./src/ ./work/src/
RUN pip install -e ./work/
