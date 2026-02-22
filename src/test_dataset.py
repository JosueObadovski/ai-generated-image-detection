from dataset import get_dataloaders

train_loader, test_loader = get_dataloaders()

print("Número de batches de treino:", len(train_loader))
print("Número de batches de teste:", len(test_loader))

# Pegando um batch para verificar formato
images, labels = next(iter(train_loader))

print("Formato das imagens:", images.shape)
print("Formato dos labels:", labels.shape)