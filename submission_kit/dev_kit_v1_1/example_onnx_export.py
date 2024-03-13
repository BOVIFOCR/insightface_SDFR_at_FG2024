# This template script should help guide the participants in the process of 
# exporting models to ONNX, when using Python and Pytorch.
# Source: 
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

import torch
from torchvision.transforms import v2 
from iresnet import iresnet50
import numpy as np
from sklearn.preprocessing import normalize as skl_normalize

onnx_model_filename = 'baseline_iresnet50.onnx'

###############################################################################
# Your model and custom preprocessing
###############################################################################
device = torch.device('cpu')

example_torch_model = iresnet50()
example_torch_model.to(device)
example_torch_model.eval() # Very important before exporting to ONNX!

test_pre_transforms = torch.nn.Sequential(
    # An example custom transformation
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
)
###############################################################################

# We wrap the preprocessing step together with the model
example_wrapped_model = torch.nn.Sequential(test_pre_transforms, 
                                            example_torch_model)

# We create an example input
example_input = torch.rand((4,3,112,112)).to(device) # Arbitrary batch-size 4
torch_out = example_wrapped_model(example_input) # Batch x Embedding-size
torch_out = torch.nn.functional.normalize(torch_out, dim=1) # Normalize emb.

# Here we export the model to ONNX.
# Once trainer, the onnx_model_filename here, as an example, is the file 
# to upload for the competition.
torch.onnx.export(example_wrapped_model,
                  example_input,
                  onnx_model_filename,
                  export_params=True,
                  opset_version=17,#latest opset supported by torch.onnx.export
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'},
                    }
                )

###############################################################################
# Sanity check to make sure the exported ONNX model provides the same outputs 
# on the same inputs as the original model.

import onnx
import onnxruntime as ort
import numpy as np

DEVICE = 'CPU'
#DEVICE = 'NVIDIA_GPU'


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() \
        if tensor.requires_grad else tensor.cpu().numpy()

# Define your margin of error
# Below are arbitrary values. Adjust them to your requirements.
rtol = 0
atol = 0

onnx_model = onnx.load(onnx_model_filename)
onnx.checker.check_model(onnx_model)

match DEVICE:
    case 'NVIDIA_GPU':
        providers=["CUDAExecutionProvider"]
        print(f"ONNX runtime currently running on NVIDIA_GPU.")
        
    case 'CPU':
        providers=["CPUExecutionProvider"]
        print(f"ONNX runtime currently running on CPU.")
    case _:
        raise ValueError(f'Unrecognized device: {DEVICE}.')

# Additional information: 
# https://onnxruntime.ai/docs/api/python/api_summary.html
ort_session = ort.InferenceSession(onnx_model_filename, providers=providers)
io_binding = ort_session.io_binding()
# OnnxRuntime will copy the data over to the compute device if 'input' is 
# consumed by nodes on the device (if CPU, no copy)
io_binding.bind_cpu_input('input', to_numpy(example_input))
io_binding.bind_output('output')
ort_session.run_with_iobinding(io_binding)  # Inferencing with inputs
ort_outs = io_binding.copy_outputs_to_cpu() # No-op if already on CPU

# We normalize the embeddings
normed_ort_outs = skl_normalize(ort_outs[0])

# compare ONNX Runtime and PyTorch results
# assert abs(actual - desired) < atol + rtol * abs(desired).
np.testing.assert_allclose(to_numpy(torch_out), 
                           normed_ort_outs, 
                           rtol=rtol,
                           atol=atol)
print("Exported model has been tested with ONNX-Runtime, "
      "and the result match within the tolerances.")
print(f"rtol: {rtol}")
print(f"atol: {atol}")