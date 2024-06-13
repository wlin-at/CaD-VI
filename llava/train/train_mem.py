# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from llava.train.train import train

#
import importlib.util
cvar_pyutils_spec = importlib.util.find_spec("cvar_pyutils")
cvar_pyutils_found = cvar_pyutils_spec is not None
if cvar_pyutils_found:
    from cvar_pyutils.debugging_tools import set_remote_debugger
    from cvar_pyutils.pytorch_utils.ddp import fix_infiniband
import os
import wandb
if __name__ == "__main__":
    if ('dccstor' in os.getcwd()) or ('dccstor' in __file__):
        print('#### CCC detected, fixing inifiniband for NCCL ...')
        fix_infiniband()
        print('#### done!')
    # os.environ['WANDB_MODE'] = 'disabled'


    # leo added
    # if True:
    #     import pydevd_pycharm
    #     pydevd_pycharm.settrace('71.184.241.45', port=80, stdoutToServer=True, stderrToServer=True, suspend=False)
    train()
