FROM jupyter/scipy-notebook:862de146632b

# install boto3, jupyter extensions and cx-oracle as jovyan
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

# switch to default user
USER $NB_UID

# Install Python dependencies from requirements.txt in advance
# Useful for development since changes in code will not trigger a layer re-build
COPY --chown=$NB_UID requirements.txt ./work/
RUN pip install --upgrade pip && \
    pip install -r ./work/requirements.txt

# install experimentation code in editable mode
COPY --chown=$NB_UID ./src/ ./work/src/
COPY --chown=$NB_UID ./setup.py ./work/
RUN pip install -e ./work/
