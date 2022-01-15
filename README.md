# Learning From the Pros: Extracting Professional Goalkeeper Technique from Broadcast Footage

### Open Source Data:
* 1v1 and Penalty images are found in the images/ folder

### Notebook Instructions:
* Clone this repo 

```console
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ python -m ipykernel install --user --name=ssac
```
* Enjoy the notebook (LearningFromThePros.ipynb) ensuring that the kernel is set to 'ssac'

### 3D Body Pose Model
We have included a notebook where you can extract the 3D body pose data from your own images! The notebook is stored in the PoseHG3D/ folder. We recommend running this notebook in Google Colab so that you can make use of their free GPU facilities. If running in Google Colab, please drag in the .py files from PoseHG3D into the Colab file system before running the model. You will also need to download the model weights at https://drive.google.com/file/d/1_2CCb_qsA1egT5c2s0ABuW3rQCDOLvPq/view


