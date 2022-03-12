# gcloud builds submit . --tag gcr.io/<your project>/ml-dataflow-image:latest

FROM python:3.7-slim

RUN apt-get update -q \
  && apt-get install --no-install-recommends -qy \
  gcc g++

# Install SDK.
RUN pip install --no-cache-dir apache-beam[gcp]==2.37.0 tensorflow-decision-forests tensorflow-model-analysis tensorflow-data-validation

# Verify that the image does not have conflicting dependencies.
RUN pip check

# Copy files from official SDK image, including script/dependencies.
COPY --from=apache/beam_python3.7_sdk:2.37.0 /opt/apache/beam /opt/apache/beam

# Set the entrypoint to Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]