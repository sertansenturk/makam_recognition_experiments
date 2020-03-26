FROM python:3.7

COPY ./requirements.txt /server/
RUN pip install -r /server/requirements.txt && \
    rm /server/requirements.txt

# RUN useradd --create-home -s /bin/bash mlflow
# USER mlflow

WORKDIR /server/
COPY ./entrypoint.sh /server/
COPY ./mlflow_test.py /server/
ENTRYPOINT ["./entrypoint.sh"]
CMD ["server"]
