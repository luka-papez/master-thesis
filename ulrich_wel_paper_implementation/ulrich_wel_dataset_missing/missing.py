should_have = []
have = []

with open('should_have.txt', 'r') as f:
  should_have = set([x.strip() for x in f.readlines()[:]])
  
with open('have.txt', 'r') as f:
  have = set([x.strip() for x in f.readlines()[:]])

print([int(x) for x in sorted(should_have - have)])
