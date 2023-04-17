This is the written process of my model training 

1. Checking up with a GPU

- As lots of individuals have known, training a model requires a GPU rather than CPU for it to be trained in a fast manner, so I ran a code to enable the free GPU that Google offered in Colaboratory 
2. Directing
- Then, I imported my Google drive to Colab, to direct to my dataset folder. As I just uploaded the folder from my local, unzipping it is neccessary. 

3. Getting the pre-made Model 

- From changing the directory to the one with the model, you can import Model from Models.(Model's name)

4.  Create dataloader

- This section includes 4 main names : train_dataset, val_dataset
                                        train_loader , val_loader

- Intialize the dataset (train and val) by transforming by resizing, tensorizing, Normalize and shuffling (of course, shuffling should apply to train data only )

5. The main part - Model training - 

- Step 1 : Evaluation metrics 
- Step 2 : Setting device for training 
- Step 3 : training the model for 10 epochs 
- Step 4 : Calculate and print training accuracy and loss 
- Step 5 : Evaluate the model on the test set
- Step 6 : Calculate and print validation accuracy and loss
- Step 7 : Print the results and save the best one in existing folder create previously 