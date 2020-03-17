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

# # install experimentation code in editable mode
# COPY ./src/ ./work/src/
# COPY ./setup.py ./work/
# RUN python3 -m pip install -e ./work/
