# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from .module import QATModule


class QuantStub(QATModule):
    r"""
    A helper QATModule doing quantize operation on input.
    """

    def forward(self, inp):
        return inp

    def forward_qat(self, inp):
        return self.apply_fakequant_with_observer(
            inp, self.act_fake_quant, self.act_observer
        )


class DequantStub(QATModule):
    r"""
    A helper QATModule doing de-quantize operation on input.
    """

    def forward(self, inp):
        return inp

    def forward_qat(self, inp):
        return inp
