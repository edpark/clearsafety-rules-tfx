FROM gcr.io/tfx-oss-public/tfx:1.6.1

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt --use-deprecated=legacy-resolver

COPY src/ src/

ENV PYTHONPATH="/pipeline:${PYTHONPATH}"