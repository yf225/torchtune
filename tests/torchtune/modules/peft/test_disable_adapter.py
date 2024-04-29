import contextlib
import torch

class TestModuleWithAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.disabled = False

    def forward(self, x):
        out = x * x
        if self.disabled:
            return out
        out = out + 1
        return out


@contextlib.contextmanager
def disable_adapter(mod):
    mod.disabled = True
    try:
        yield
    finally:
        mod.disabled = False


mod = TestModuleWithAdapter()
mod = torch.compile(mod)
x = torch.randn(4, 4)
out_enable = mod(x)
with disable_adapter(mod):
    out_disable = mod(x)
assert torch.allclose(out_enable, x * x + 1)
assert torch.allclose(out_disable, x * x)
