import torch
import numpy as np
import tensorrt
import io
import PIL
import logging
import torchvision.transforms.functional as F


ImageTypes = (PIL.Image.Image, np.ndarray, torch.Tensor)
ImageExtensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")


class cudaArrayInterface:
    """
    Exposes __cuda_array_interface__ - typically used as a temporary view into a larger buffer
    https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    """

    def __init__(self, data, shape, dtype=np.float32):
        if dtype == np.float32:
            typestr = "f4"
        elif dtype == np.float64:
            typestr = "f8"
        elif dtype == np.float16:
            typestr = "f2"
        else:
            raise RuntimeError(f"unsupported dtype:  {dtype}")

        self.__cuda_array_interface__ = {
            "data": (data, False),  # R/W
            "shape": shape,
            "typestr": typestr,
            "version": 3,
        }


torch_dtype_dict = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}


def torch_dtype(dtype):
    """
    Convert numpy.dtype or str to torch.dtype
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    elif not isinstance(dtype, type):
        # from np.dtype() (not a built-in np.float32, ect)
        torch_dtype = torch_dtype_dict.get(str(dtype))

        if torch_dtype is None:
            raise ValueError("unknown dtype {dtype}  (type={type(dtype)}")

        return torch_dtype

    if dtype == np.float32:
        return torch.float32
    elif dtype == np.float64:
        return torch.float64
    elif dtype == np.int8:
        return torch.int8
    elif dtype == np.int16:
        return torch.int16
    elif dtype == np.int32:
        return torch.int32
    elif dtype == np.int64:
        return torch.int64
    elif dtype == np.uint8:
        return torch.uint8
    elif dtype == np.uint16:
        return torch.uint16
    elif dtype == np.uint32:
        return torch.uint32
    elif dtype == np.uint64:
        return torch.uint64
    elif dtype == np.complex64:
        return torch.complex64
    elif dtype == np.complex128:
        return torch.complex128
    elif dtype == np.bool_:
        return torch.bool

    raise ValueError("unknown dtype {dtype}  (type={type(dtype)}")


def convert_dtype(dtype, to="np"):
    """
    Convert a string, numpy type, or torch.dtype to either numpy or PyTorch
    """
    if dtype is None:
        return None

    if to == "pt":
        return torch_dtype(dtype)
    elif to == "np":
        if isinstance(dtype, type):
            return dtype
        elif isinstance(dtype, torch.dtype):
            return np.dtype(str(dtype).split(".")[-1])  # remove the torch.* prefix
        else:
            return np.dtype(dtype)

    raise TypeError(
        f"expected dtype as a string, type, or torch.dtype (was {type(dtype)}) and with to='np' or to='pt' (was {to})"
    )


def convert_tensor(tensor, return_tensors="pt", device=None, dtype=None, **kwargs):
    """
    Convert tensors between numpy/torch/ect
    """
    if tensor is None:
        return None

    if isinstance(return_tensors, type):
        if return_tensors == np.ndarray:
            return_tensors = "np"
        elif return_tensors == torch.Tensor:
            return_tensors = "pt"
        elif return_tensors == list:
            return_tensors = "list"
        else:
            raise TypeError(f"expected return_tensors as np.ndarray, torch.Tensor, or list (was {return_tensors})")

    dtype = convert_dtype(dtype, to=return_tensors)

    if isinstance(tensor, np.ndarray):
        if return_tensors == "np":  # np->np
            if dtype:
                tensor = tensor.astype(dtype=convert_dtype(dtype, to="np"), copy=False)
            return tensor
        elif return_tensors == "pt":  # np->pt
            return torch.from_numpy(tensor).to(device=device, dtype=convert_dtype(dtype, to="pt"), **kwargs)
        elif return_tensors == "list":  # np->list
            return tensor.tolist()
    elif isinstance(tensor, torch.Tensor):
        if return_tensors == "np":  # pt->np
            if dtype:
                tensor = tensor.type(dtype=convert_dtype(dtype, to="pt"))
            return tensor.detach().cpu().numpy()
        elif return_tensors == "pt":  # pt->pt
            if device is not None or dtype is not None:
                return tensor.to(device=device, dtype=convert_dtype(dtype, to="pt"), **kwargs)
            else:
                return tensor
        elif return_tensors == "list":
            return tensor.tolist()
    elif isinstance(tensor, list):
        if return_tensors == "np":
            return np.asarray(tensor, dtype=dtype)
        elif return_tensors == "pt":
            return torch.as_tensor(tensor, dtype=dtype, device=device)
        elif return_tensors == "list":
            return tensor

    raise ValueError(f"unsupported tensor input/output type (in={type(tensor)} out={return_tensors})")


class AttributeDict(dict):
    """
    A dict where keys are available as attributes:

      https://stackoverflow.com/a/14620633

    So you can do things like:

      x = AttributeDict(a=1, b=2, c=3)
      x.d = x.c - x['b']
      x['e'] = 'abc'

    This is using the __getattr__ / __setattr__ implementation
    (as opposed to the more concise original commented out below)
    because of memory leaks encountered without it:

      https://bugs.python.org/issue1469629

    TODO - rename this to ConfigDict or NamedDict?
    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, value):
        self.__dict__ = value


def trt_model_filename(model, suffix=None, ext=".pt"):
    """
    Returns a filename for a TensorRT engine that includes the TensorRT version and CUDA device SM.
    This is used so that the cached TensorRT engines only get used with the same version of TensorRT and GPU.
    """
    model = model.replace("/", "-").replace("@", "-")
    trt_version = tensorrt.__version__.replace(".", "")
    cuda_sm = torch.cuda.get_device_capability()
    cuda_sm = f"sm{cuda_sm[0]}{cuda_sm[1]}"
    suffix = f"_{suffix}" if suffix else ""
    return f"{model}_trt-{trt_version}_{cuda_sm}{suffix}{ext}"


def torch_image(image, dtype=None, device=None):
    """
    Convert the image to a type that is compatible with PyTorch ``(torch.Tensor, ndarray, PIL.Image)``
    """
    if not isinstance(image, ImageTypes):
        raise TypeError(f"expected an image of type {ImageTypes} (was {type(image)})")

    if isinstance(image, (PIL.Image.Image, np.ndarray)):
        image = F.to_tensor(image)

    return image.to(dtype=dtype, device=device)
