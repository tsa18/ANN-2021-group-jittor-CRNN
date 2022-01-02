from jittor.utils.pytorch_converter import convert

pytorch_code="""
import os
import glob

import torch
if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""

jittor_code = convert(pytorch_code)
print(jittor_code)