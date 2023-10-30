
import torch

print(torch.__path__)

x = torch.tensor([[9223372036854775809, 2, 3], [4, 5, 6]], dtype=torch.uint64)
y = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.uint64)
z = x

print(z)


x = torch.tensor([[922337203685477, 2, 3], [4, 5, 6]], dtype=torch.float8_e5m2)
x = torch.tensor([[922337203685477, 2, 3], [4, 5, 6]], dtype=torch.field64)
y = torch.tensor([[922337203685477, 2, 3], [4, 5, 6]], dtype=torch.field64)

z = torch.add(x, y, alpha=2)

print(z)

# m = torch.nn.MyFUNC(inplace=True)



# x = m(x)
# print(x.cpu())


# x = x.cuda()
# x = m(x) #now in GPU
# print(x.cpu())

# c = x.tolist()

# d = torch.tensor(c, dtype=torch.uint64)

# print(d)



