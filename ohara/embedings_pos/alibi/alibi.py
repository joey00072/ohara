
import torch


def get_alibi_mask(number_of_heads,max_seq_len):
    """
    https://arxiv.org/abs/2108.12409 \\
    ref: https://www.youtube.com/watch?v=Pp61ShI9VGc \\
    Add this with casulmask then use that mask for efficeny \\
    example:
    ```
    >> mask = casual_mask + get_alibi_mask(number_of_heads,max_seq_len)
    >> wei = softmax(q@k.transpose(-1,-2) + mask[:,:T,:T])
    ```
    
    """
    
    nh = number_of_heads
    n = max_seq_len
    
    rows = torch.arange(1, n+1).view(1, -1, 1)
    cols = torch.arange(n).view(1, 1, -1)

    matrix = (rows - cols)
    matrix = torch.where(matrix < 1, (torch.zeros_like(matrix)), matrix)
    
    matrix = matrix.expand(nh, -1, -1)* -1
    m = 1/2**(torch.arange(1,nh+1)/(nh/8))
    return matrix * m.view(nh,1,1)

if __name__=="__main__":
    print(get_alibi_mask(16,5))
