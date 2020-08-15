ARG SPHINX_VERSION
FROM sphinxdoc/sphinx:${SPHINX_VERSION}

ENV SPHINX_OPTS=

RUN useradd -ms /bin/bash sphinxuser && \
    mkdir /makam_recognition_experiments && \
    chown -R sphinxuser /makam_recognition_experiments && \
    pip install sphinx_rtd_theme
USER sphinxuser

WORKDIR /makam_recognition_experiments/docs

CMD make html SPHINXOPTS:=$SPHINX_OPTS
