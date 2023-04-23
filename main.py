from enrico_utils.get_data import get_dataloader

(train_loader, val_loader, test_loader), weights = get_dataloader("enrico_corpus")


print(train_loader)