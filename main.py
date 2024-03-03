import torch
from torch import nn
from helper_classes.data_prep import create_directories_and_download_datasets, create_datasets_and_dataloaders
from helper_classes.engine import train
from helper_classes.model import Model
from helper_classes.utils import plot_loss_curves
from pathlib import Path

if __name__ == "__main__":
    #Create the folder structure, directories and download the datasets
    create_directories_and_download_datasets()

    #Create and return the dataloaders for test and train
    train_dataloader, test_dataloader = create_datasets_and_dataloaders()

    # Get the model to train
    model = Model()

    # Create a path to save our trained model later
    model_save_path = Path("model/")

    # Code to start the model traiing and testing phases:
    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 40

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # Start the timer
    from timeit import default_timer as timer
    start_time = timer()

    # Train model_1
    trained_model, model_1_results = train(model=model,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=NUM_EPOCHS)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")

    # Check the model_0_results keys
    model_1_results.keys()

    #plot lose curve
    plot_loss_curves(model_1_results)

    #save model
    torch.save(obj=trained_model.state_dict(),
               f=model_save_path)