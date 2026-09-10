"""Retain immutable inference parameter storage across synchronous group offload.

Avoid replacing shared mmap CPU parameters with a private CUDA-to-CPU copy on
both ranks. Buffers retain Diffusers' normal copy-back semantics. Not for training.
"""
import torch

def install():
    from diffusers.hooks.group_offloading import ModuleGroup
    if getattr(ModuleGroup, '_sol_retention_installed', False):
        return
    original_onload = ModuleGroup._onload_from_memory
    original_offload = ModuleGroup._offload_to_memory
    def onload(group):
        if group.stream is None and group.offload_device == torch.device('cpu'):
            parameters = list(group.parameters)
            for module in group.modules:
                parameters.extend(module.parameters())
            unique = list(dict.fromkeys(parameters))
            if hasattr(group, '_sol_cpu_parameters'):
                raise RuntimeError('nested group onload without offload')
            if any(type(p) is not torch.nn.Parameter or p.device.type != 'cpu' for p in unique):
                raise RuntimeError('retention requires ordinary CPU parameters')
            if torch.is_grad_enabled():
                raise RuntimeError('CPU retention is for inference only')
            group._sol_cpu_parameters = {p:p.data for p in unique}
        return original_onload(group)
    def offload(group):
        retained = getattr(group, '_sol_cpu_parameters', None)
        if retained is not None:
            for parameter, cpu_data in retained.items():
                parameter.data = cpu_data
            del group._sol_cpu_parameters
        return original_offload(group)
    ModuleGroup._onload_from_memory = onload
    ModuleGroup._offload_to_memory = offload
    ModuleGroup._sol_retention_installed = True

@torch.inference_mode()
def verify(device):
    from diffusers.hooks import apply_group_offloading
    model=torch.nn.Sequential(torch.nn.Linear(32,64),torch.nn.SiLU(),torch.nn.Linear(64,32)).to(dtype=torch.bfloat16)
    original={p:p.data_ptr() for p in model.parameters()}
    clone=torch.nn.Sequential(torch.nn.Linear(32,64),torch.nn.SiLU(),torch.nn.Linear(64,32)).to(device=device,dtype=torch.bfloat16)
    clone.load_state_dict(model.state_dict())
    inputs=torch.randn(4,32,device=device,dtype=torch.bfloat16)
    expected=clone(inputs)
    apply_group_offloading(model,onload_device=device,offload_device=torch.device('cpu'),offload_type='leaf_level',use_stream=False)
    for _ in range(2):
        actual=model(inputs)
        assert torch.equal(actual,expected)
        assert all(p.device.type=='cpu' and p.data_ptr()==pointer for p,pointer in original.items())
    del model,clone,inputs,actual,expected
    torch.cuda.empty_cache()
