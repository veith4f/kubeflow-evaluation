# digits-recognizer
this is an evaluation of kubeflow. creates a digit recognizer model working on images. tested to work on kubeflow v1.9.1 with mnist.npz.

## deploying kubeflow
step 1
```bash
git clone git@github.com:kubeflow/manifests.git kubeflow-manifests
cd kubeflow-manifests
git checkout -b v1.9.1-branch origin/v1.9.1-branch
while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do echo "Retrying to apply resources"; sleep 20; done
```
step 2
```bash
git clone git@github.com:veith4f/digits-recognizer.git
cd digits-recognizer
kubectl create ns kubeflow-user-example-com # namespace of default user
kubectl apply -f configs/mysql-peer-auth.yml # mysql database must not do mtls
kubectl apply -f configs/global-allow-all.yml # make the istio-system/global-deny-all a global allow-all
kubectl apply -f configs/access_kfp_from_jupyter_notebook.yaml # jupyter notebooks of default user (namespace kubeflow-user-example-com) can create pipelines
kubectl apply -f configs/minio-kserve-secret.yml # kserve can authenticate to built-in minio
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```
then open the dashboard on http://localhost:8000 and log in as user@example.com/12341234

## notebooks 
- create a new notebook and be sure to add `configuration->Allow access to kubeflow pipelines`.
- wait for the notebook to become available, then connect.
- open console in notebook, then
```
pip install minio
```
- drag/drop the .ipynb files from notebooks folder into the files section on left side of jupyter instance.

### exploration
- digits_recognize.ipynb is a classic ai hello world example training a model to classify images showing digits.
- digits_generate.ipynb is a slightly more involved example that leverages a generative adverserial network model to generate new images showing digits.

### pipelines
set correct minio ip in each dsl.component before executing pipeline
```bash
kubectl get -n kubeflow svc/minio-service # get cluster-ip and set accordingly in notebook
```
- digits_recognize_pipeline.ipynb creates an experiment and runs pipelines as part of that experiment.
- you can also compile the pipeline into a yaml in Pipeline Intermediate Representation (IR) which can be executed on Google Cloud.

## use prediction model served with kserve
- when the pipeline runs it will serve a model with kserve from namespace kubeflow-user-example-com
- forward this model to localhost:8000
```bash
kubectl port-forward -n kubeflow-user-example-com svc/digits-recognizer-predictor-00001-private 8000:80
```
- prepare python venv and send digit.png to endpoint for prediction
```bash
python3.12 -m venv .
source bin/activate
pip3 install -r requirements.txt
python scripts/predict_image.py assets/digit.png
```
results should be something like
```yaml
{'predictions': [[1.33714252e-07, 1.17570687e-06, 1.03824357e-06, 0.0074280994, 1.30658311e-08, 0.984939337, 5.25189944e-06, 1.1648237e-05, 1.52014836e-05, 0.00759810442]]}
```
