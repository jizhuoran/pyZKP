
import torch

print(torch.__path__)

x = torch.tensor([[922337203685498, 2, 3], [4, 5, 6]], dtype=torch.big_integer)
y = torch.tensor([[922337203685477, 2, 3, 10], [4, 5, 6, 8]], dtype=torch.big_integer)

print(x)
print(y)


print(x[1])

# # x = torch.tensor([[9223372036854775809, 2, 3], [4, 5, 6]], dtype=torch.uint64)
# # y = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.uint64)
# # z = x

# # x = torch.tensor([[922337203685477, 2, 3], [4, 5, 6]], dtype=torch.float8_e5m2)
# y = torch.tensor([[922337203685477, 2, 3, 10], [4, 5, 6, 8]], dtype=torch.big_integer)
# # x = torch.tensor([[922337203685477, 2, 3], [4, 5, 6]], dtype=torch.field64)

# print(y)

# # z = torch.add(x, y, alpha=2)

# # print(z) 

# # m = torch.nn.MyFUNC(inplace=True)



# # x = m(x)
# # print(x.cpu())


# # x = x.cuda()
# # x = m(x) #now in GPU
# # print(x.cpu())

# # c = x.tolist()

# # d = torch.tensor(c, dtype=torch.uint64)

# # print(d)



