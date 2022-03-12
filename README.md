# Rules in TFDF

This is a fork of the end-to-end [MLOps process](https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf) as implemented in Google's [mlops-with-vertex-ai example](https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai).

One significant difference is the use of Tensorflow's [Decision Forests](https://www.tensorflow.org/decision_forests) to overfit (memorize) a table of rules. More information can be found [here](https://medium.com/@shakabrah/tensorflow-tfx-and-decision-forests-on-google-vertex-ai-625c03784410).


## Getting started

1. Install Conda
2. Create a new Conda environment (eg. `conda create -n tfdf-rules python=3.8`)
3. Activate the new Conda environment: `conda activate tfdf-rules`
4. Clone the repository:
    ```
    git clone https://github.com/edpark/tfdf-rules-tfx.git
    cd tfdf-rules-tfx
    ```
5. Install the required Python packages:
    ```
    pip install -r requirements.txt --use-deprecated=legacy-resolver
    ```
    NOTE: 
    You may need to install gcc to build some of the requirements:
    `sudo apt-get update -q && sudo apt-get install --no-install-recommends -qy gcc g++`
6. Register this environment with the Jupyter plugin: `python -m ipykernel install --user --name=tfdf-rules`
7. Sign up for a Google Cloud account: https://cloud.google.com/free
8. Create a project in Google Cloud and a Cloud Storage bucket in the `us-central1` region (NOT the `Multi-region` option)
9. Upgrade the `gcloud` components:
   ```
   sudo apt-get install google-cloud-sdk
   gcloud components update
   ```
10. Run through the notebooks in order (01 - 08)


## Notebook Overview
### 01-dataset-management
This notebook has been pared down from the original which used BigQuery to create a dataset from the taxi data.
Sections '2. Create data for the ML task' and '3. Generate raw data schema' have been removed as they do not apply to TFDF.

### 02-experimentation
Uses custom training to train a TFDF classifier.

### 03-training-formalization
Runs each of the TFX pipeline steps using the TFX `InteractiveContext`.

### 04-pipeline-deployment
Tests, deploys, and runs the TFX pipeline on Vertex AI Pipelines.

### 05-continuous-training
Creates and uses Cloud Functions and Cloud Pub/Sub to trigger pipeline execution.

### 06-model-deployment
Executes a CI/CD routine to test and deploy the trained model from the previous notebook to a Vertex AI Endpoint.

### 07-prediction-serving
Use the deployed model for online prediction.


## Notes and Errata

1. Needed to add `custom-online-prediction@XXXXXXXX-tp.iam.gserviceaccount.com` to the list of Principals with access to the bucket being used for the pipeline runs
otherwise a Permission denied error occurs when trying to deploy the model to an endpoint:
```
"custom-online-prediction@X-tp.iam.gserviceaccount.com does not have storage.objects.get access to the Google Cloud Storage object."
E tensorflow_serving/sources/storage_path/file_system_storage_path_source.cc:365] FileSystemStoragePathSource encountered a filesystem access error: Could not find base path gs://tfdf-rules-dev/tfdf-rules/model_registry/tfdf-rules-classifier-v01 for servable tfdf-rules-classifier-v01 with error Permission denied: Error executing an HTTP request: HTTP response code 403 when reading metadata of gs://tfdf-rules-dev/tfdf-rules/model_registry/tfdf-rules-classifier-v01
```

2. The Evaluator component works when pipeline is run locally but encounters an error when running in Vertex AI and is disabled. There was some discussion on the forums on this issue but is still unresolved: https://discuss.tensorflow.org/t/tensorflow-decision-forests-with-tfx-model-serving-and-evaluation/2137/49

3. A custom container was built for the Evaluator component to run on (under `build-dataflow/`) as the default one used with Beam is missing the necessary dependencies. This image is set under the `CUSTOM_DATAFLOW_IMAGE_URI` environment variable.

4. The default Tensorflow Serving image is missing the dependencies necessary to use TFDF as outlined [here](https://github.com/tensorflow/decision-forests/blob/main/documentation/tensorflow_serving.md) so the approach taken was to use a container built by the ml6team that includes the necessary Ops for TFDF: https://hub.docker.com/repository/docker/ml6team/tf-serving-tfdf. The ml6team's blog post is [here](https://blog.ml6.eu/serving-decision-forests-with-tensorflow-b447ea4fc81c). As of TFDF version 0.2.3, the team has released a Tensorflow Serving binary with the necessary TFDF Ops but I haven't figured out how to use this binary yet.

5. These were the commands necessary to run in order to create a Vertex AI service account and give it the required permissions:
```
$ gcloud iam service-accounts create vertex-ai --description="Vertex AI Service Acct" --display-name="vertex-ai" --project=<your project>
$ gcloud projects add-iam-policy-binding <your project> --member="serviceAccount:vertex-ai@<your project>.iam.gserviceaccount.com" --role="roles/aiplatform.user"
$ gsutil iam ch serviceAccount:vertex-ai@<your project>.iam.gserviceaccount.com:legacyBucketWriter gs://artifacts.<your project>.appspot.com
$ gsutil iam ch serviceAccount:vertex-ai@<your project>.iam.gserviceaccount.com:legacyBucketWriter gs://us.artifacts.<your project>.appspot.com
$ gsutil iam ch serviceAccount:vertex-ai@<your project>.iam.gserviceaccount.com:roles/storage.objectCreator gs://<your bucket>
$ gsutil iam ch serviceAccount:vertex-ai@<your project>.iam.gserviceaccount.com:roles/storage.objectViewer gs://<your bucket>
$ gsutil iam ch serviceAccount:vertex-ai@<your project>.iam.gserviceaccount.com:roles/storage.legacyObjectOwner gs://<your bucket>
$ gsutil iam ch serviceAccount:vertex-ai@<your project>.iam.gserviceaccount.com:roles/storage.legacyBucketOwner gs://<your bucket>
$ gcloud iam service-accounts add-iam-policy-binding vertex-ai@<your project>.iam.gserviceaccount.com --member="user:<your email>" --role="roles/iam.serviceAccountUser"

Also:
# Grant default service account to act as a Service Account User role on this new vertex-ai service account (https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline#before_you_begin)
# Needs Storage Object Viewer/Creator + Storage Admin on bucket where pipeline files are being written to
# Needs Dataflow Worker perms
```

## Support

[Discussion Forum on Github](https://github.com/edpark/tfdf-rules-tfx/discussions)

[Issues on Github](https://github.com/edpark/tfdf-rules-tfx/issues)