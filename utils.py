# aquire the embedding of the image by concating a submodule      
t = torch.nn.Sequential(*(list(net.features)+list(net.embedding)))
output = t(val_image)
print(output)
print(output.shape)