# DETRfastapi
Object Detection API powered by DETR and FastAPI

# How to run the prediciton API:

A. Using Docker [recommended]:

1. Build Docker image from the Dockerfile included in the repo.
   ```
   cd fastapi
   docker build -t fastapi .
   ```
2. Run the Docker image in detached mode.
   ```
   docker run --rm -it --name fastapicontainer -p 90:80 fastapi
   ```
 Now the API endpoint should be live at http://127.0.0.1:90

B. Using venv / normal python environment:
   1. Install the necessary python packages.
      ```
      cd fastapi
      pip install -r requirements.txt
      ```
   2. Start the server.
      ```
      cd fastapi/app
      uvicorn main:app --reload
      ```
  Now the API endpoint should be live at http://localhost:8000

# How to store big model files using DVC + Azure:

1. Initialize dvc repository:
   ```
   dvc init
   ```
2. Add `.pth` files under DVC version control:
   ```
   dvc add fastapi/resnet50-19c8e357.pth
   git add fastapi/resnet50-19c8e357.pth.dvc fastapi/.gitignore
   dvc add fastapi/app/detr-r50-e632da11.pth
   git add fastapi/app/detr-r50-e632da11.pth.dvc fastapi/app/.gitignore
   ```
3. Set up remote Azure Cloud:
   ```
   dvc remote add -d fastapi azure://fastapi/storage
   dvc remote modify --local fastapi connection_string "XXXXX" # Get XXXXX from the proper Azure BlobStorage account. [in this case, nvi0426asa -> Access Keys] 
   ```
4. Push the data files in Azure Cloud:
   ```
   dvc push fastapi/resnet50-19c8e357.pth.dvc
   dvc push fastapi/app/detr-r50-e632da11.pth.dvc
   ```
 Now these 2 model files will be tracked by DVC and can be accessed any time by performing `dvc pull fastapi/resnet50-19c8e357.pth.dvc` and `dvc pull fastapi/app/detr-r50-e632da11.pth.dvc`
