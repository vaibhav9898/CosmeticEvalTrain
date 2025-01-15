There are three bash scripts in this project for training and Inference.

-train.sh is for training single model on the data
-inference.sh is for infering single model on the data
-multiple_voter.sh is for infering all the trained models on the data

Steps:

0. Activate symmetry environment on DGX using conda activate symmetry. This contains all the dependences for the Project
1. Create a TSV train and test File similar to the TSV File as located in TSV_Files folder
2. This File will have image paths, keypoints and class of the image
3. Now train a resnet, swin transformer, mobileViT and densenet using the train.sh script by changing the model name
4. Once the models are trained, you can check the model's output using inference.sh
5. Once satisfied with the model's output individually, set the checkpoint paths and run the multiple_voter.sh, this will create a joint inference with all the models.

NOTE : During the use of model through GUI, an option is given to save the data for training to the user. This will generate a CSV File, which should be converted to appropriate format for training the models
NOTE : Repo can have minor errors like Path not set, Image not found etc. Check out how to set path using sys python package and check if your CSV File loads the images correctly
NOTE : It is advisable to check the images once they are loaded during training after augmentation, for sanity purpose. Ensure that keypoints and Images are consistent


### FOR INFERENCE TO ABHIJEET

keypoints -> Foo1(Image)
# Process Keypoints -> manual shifting -> Final
class value -> Foo2(Image, Final_Keypoints)


Checkpoints are available at https://iitbacin-my.sharepoint.com/:f:/g/personal/23m1062_iitb_ac_in/EmwTlpzi5yxNsbR58JPiJtUBMmhg0yT6QlzPki8_tZjC1A?e=VQKHFX

Video recording of explaination is available here https://iitbacin-my.sharepoint.com/:v:/g/personal/23m1062_iitb_ac_in/EaFpWNuLmXlMmzzwGaKholAB0aQbbtCqVLwyjyoWlo0nCQ?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=DWraQF
