## Alibi
This is so simple I love it

![Alt text](image.png)
This is lower tringuer matrix with with every token in past multiplied by -1

### Imp
you multiplay each each by no $m$ and it is diffent for each head.

1. **When `seq_len` is 8**:
   The series is $\frac{1}{2^n}$ where $ n $ ranges from 1 to 8. In LaTeX, this can be written as:
$\frac{1}{2^1}, \frac{1}{2^2}, \frac{1}{2^3}, \ldots, \frac{1}{2^8}$

2. **When `seq_len` is 16**:
   The series is $ \frac{1}{2^n} $ where $ n $ starts at 0.5 and increases in increments of 0.5 up to 8. In LaTeX, this can be represented as:
$\frac{1}{2^{0.5}}, \frac{1}{2^1}, \frac{1}{2^{1.5}}, \ldots, \frac{1}{2^8}$


explained in code if you like me stranger

```python
s = torch.arange(1,nh+1)
p = s/(nh/8)
m = 1/2**p
```

